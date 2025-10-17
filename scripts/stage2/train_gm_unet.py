#!/usr/bin/env python
# coding=utf-8
"""Stage 2: Fine-tune Stable Diffusion UNet to predict HDR guidance maps (GM)."""
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel, PNDMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import torchvision
import cv2

if is_wandb_available():
    import wandb

# from pipline_stable_diffusion_concat import StableDiffusionPipeline
from gm_diffusion.pipelines import StableDiffusionGMPipeline

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# DATASET_NAME_MAPPING = {
#     "lambdalabs/naruto-blip-captions": ("image_patch", "text"),
# }

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def save_model_card(
    args,
    repo_id: str,
    images: list = None,
    repo_folder: str = None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(vae, text_encoder, tokenizer, gm_unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")
    num_train_timesteps = 50

    val_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if args.scheduler_config:
        pndm_scheduler = PNDMScheduler.from_config(args.scheduler_config)
    else:
        pndm_scheduler = PNDMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )

    gm_pipeline = StableDiffusionGMPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(gm_unet),
        safety_checker=None,
        scheduler=pndm_scheduler,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    gm_pipeline = gm_pipeline.to(accelerator.device)
    gm_pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        gm_pipeline.enable_xformers_memory_efficient_attention()

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.validation_prompts:
        validation_prompts = list(args.validation_prompts)
    elif args.validation_prompt_file:
        prompt_path = Path(args.validation_prompt_file)
        if not prompt_path.exists():
            raise ValueError(f"Validation prompt file not found: {prompt_path}")
        validation_prompts = [line.strip() for line in prompt_path.read_text().splitlines() if line.strip()]
    else:
        raise ValueError("Please provide --validation_prompts or --validation_prompt_file for validation logging.")

    if not args.validation_image_dir:
        raise ValueError("--validation_image_dir is required in order to load SDR reference images for validation.")
    image_dir = Path(args.validation_image_dir)
    if not image_dir.exists():
        raise ValueError(f"Validation image directory not found: {image_dir}")
    validation_images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]
    )
    if len(validation_images) < len(validation_prompts):
        raise ValueError("Number of validation images must match or exceed number of prompts.")

    images = []
    for i, prompt in enumerate(validation_prompts):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            eval_sdr_input_path = validation_images[i]
            logger.info(f"eval_sdr_input_path: {eval_sdr_input_path}")
            logger.info(f"validation_prompt: {prompt}")
            eval_sdr_input = Image.open(eval_sdr_input_path)

            sdr_image = val_transforms(eval_sdr_input)
            sdr_min = float(sdr_image.min().item())
            sdr_max = float(sdr_image.max().item())
            if not (-1 <= sdr_min <= sdr_max <= 1):
                raise ValueError("Validation SDR images must be normalized into [-1, 1] after transforms.")
            sdr_image = sdr_image.unsqueeze(0).to(accelerator.device)
            sdr_latent = vae.encode(sdr_image.to(weight_dtype)).latent_dist.sample()
            sdr_latent = sdr_latent * vae.config.scaling_factor

            gm_latent = gm_pipeline(
                sdr_latent,
                prompt,
                num_inference_steps=num_train_timesteps - 1,
                generator=generator,
                output_type="latent",
            ).images[0]

            gm_latent = gm_latent.unsqueeze(0)
            sdr_latent = sdr_latent / vae.config.scaling_factor
            gm_latent = gm_latent / vae.config.scaling_factor

            sdr_image = vae.decode(sdr_latent, return_dict=False)[0]
            sdr_image = (sdr_image / 2 + 0.5).clamp(0, 1)
            sdr_image = sdr_image.cpu().permute(0, 2, 3, 1).float().numpy()

            gm_image = vae.decode(gm_latent, return_dict=False)[0]
            gm_image = (gm_image / 2 + 0.5).clamp(0, 1)
            gm_image = gm_image.cpu().permute(0, 2, 3, 1).float().numpy()

            gm_sdr_concat = np.concatenate((sdr_image, gm_image), axis=-2)
            images.append(gm_sdr_concat)

    # Log images to TensorBoard or WandB
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    
    del gm_pipeline
    torch.cuda.empty_cache()

    return images



from torchvision import transforms
import random
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--scheduler_config",
        type=str,
        default=None,
        help="Optional path to a scheduler config directory to override the default.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_metadata",
        type=str,
        nargs='+',
        default=None,
        help="One or more parquet files containing SDR/GM metadata.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for Hugging Face datasets.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_prompt_file",
        type=str,
        default=None,
        help="Optional path to a text file containing one validation prompt per line.",
    )
    parser.add_argument(
        "--validation_image_dir",
        type=str,
        default=None,
        help="Directory with SDR reference images used during validation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--use_x0_conditioning",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dream_training",
        action="store_true",
        help=(
            "Use the DREAM training method, which makes training more efficient and accurate at the ",
            "expense of doing an extra forward pass. See: https://arxiv.org/abs/2312.00210",
        ),
    )
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="Dream detail preservation factor p (should be greater than 0; default=1.0, as suggested in the paper)",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def _replace_unet_conv_in(unet_model):
    # replace the first layer to accept 8 in_channels
    _weight = unet_model.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = unet_model.conv_in.bias.clone()  # [320]
    _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
    # half the activation magnitude
    _weight *= 0.5
    # new conv_in channel
    _n_convin_out_channel = unet_model.conv_in.out_channels
    _new_conv_in = torch.nn.Conv2d(
        8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = torch.nn.Parameter(_weight)
    _new_conv_in.bias = torch.nn.Parameter(_bias)
    unet_model.conv_in = _new_conv_in
    logging.info("Unet conv_in layer is replaced")
    # replace config
    unet_model.config["in_channels"] = 8
    logging.info("Unet config is updated")
    return unet_model

# def _replace_unet_conv_out(unet_model):
#     # replace the first layer to accept 8 in_channels
#     _weight = unet_model.conv_out.weight.clone()  # [320, 4, 3, 3]
#     _bias = unet_model.conv_out.bias.clone()  # [320]
#     _bias = _bias.repeat((2))  # Keep selected channel(s)
#     _weight = _weight.repeat((2, 1, 1, 1))  # Keep selected channel(s)
#     # half the activation magnitude
#     _weight *= 0.5
#     # new conv_in channel
#     _new_conv_out = torch.nn.Conv2d(
#         320, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
#     )
#     _new_conv_out.weight = torch.nn.Parameter(_weight)
#     _new_conv_out.bias = torch.nn.Parameter(_bias)
#     unet_model.conv_out = _new_conv_out
    
#     logging.info("Unet conv_out layer is replaced")
#     unet_model.config["out_channels"] = 8
#     logging.info("Unet config is updated")
#     return unet_model


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    gm_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )
    gm_unet = _replace_unet_conv_in(gm_unet)
    # gm_unet = _replace_unet_conv_out(gm_unet)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    gm_unet.requires_grad_(True)
    gm_unet.conv_in.requires_grad_(True)
    # gm_unet.conv_out.requires_grad_(True)
    
    # lora gm_unet
    # gm_unet_lora_config = LoraConfig(
    #     r=128,
    #     lora_alpha=128,
    #     init_lora_weights="gaussian",
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    # )
    # gm_unet.add_adapter(gm_unet_lora_config)
    print_trainable_parameters(gm_unet)

    # Create EMA for the unet.
    if args.use_ema:
        ema_gm_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_gm_unet = EMAModel(
            ema_gm_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_gm_unet.config,
            foreach=args.foreach_ema,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            gm_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_gm_unet.save_pretrained(os.path.join(output_dir, "gm_unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel, foreach=args.foreach_ema
                )
                ema_unet.load_state_dict(load_model.state_dict())
                if args.offload_ema:
                    ema_unet.pin_memory()
                else:
                    ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        # unet.enable_gradient_checkpointing()
        gm_unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # sdr_lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    # gm_lora_layers = filter(lambda p: p.requires_grad, gm_unet.parameters())

    gm_optimizer = optimizer_cls(
        list(gm_unet.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # import glob

    if not args.train_metadata:
        raise ValueError("--train_metadata must point to at least one parquet file.")
    data_files = args.train_metadata if len(args.train_metadata) > 1 else args.train_metadata[0]
    train_dataset = load_dataset('parquet', data_files=data_files, split='train', cache_dir=args.dataset_cache_dir)

    from PIL import Image
    import io
    import torchvision.transforms as transforms

    def preprocess_train(examples):
        """
        Preprocess training data by loading paired images and gainmaps from byte arrays, applying paired transformations,
        and converting to tensors.

        Args:
            examples (dict): Batch of examples containing 'input_image', 'output_image', and 'text'.

        Returns:
            dict: Processed examples with 'pixel_values' and 'gainmap_values'.
        """
        # Extract SDR images, GM images, and captions from examples
        sdr_paths = examples["sdr"]
        gm_images = examples["gainmap"]
        captions = examples["text"]

        # Define the final transformations to apply
        train_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Process each pair of SDR and GM images
        processed_pixel_values = []
        processed_gainmap_values = []

        for sdr_path, gm_img_bytes in zip(sdr_paths, gm_images):
            # Convert byte arrays to PIL Images
            sdr_image = Image.open(sdr_path).convert("RGB")
            gm_image = Image.open(io.BytesIO(gm_img_bytes)).convert("RGB")

            # Apply transformations
            pixel_tensor = train_transforms(sdr_image)
            gainmap_tensor = train_transforms(gm_image)

            assert pixel_tensor.shape == gainmap_tensor.shape, "SDR image and GM image shapes don't match"

            processed_pixel_values.append(pixel_tensor)
            processed_gainmap_values.append(gainmap_tensor)

        # add tokenizer
        # from transformers import CLIPTextModel, CLIPTokenizer
        # tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        # print(captions)
        # try:
        # except:
        #     print("caption is not righr!")
        # shit_captions = ["shit"]*len(captions)
        tokenized_captions = tokenizer(
            captions, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=77, 
            truncation=True
        )

        # Assign processed tensors back to examples
        examples["pixel_values"] = processed_pixel_values
        examples["gainmap_values"] = processed_gainmap_values
        examples["input_ids"] = tokenized_captions["input_ids"]
        examples["attention_mask"] = tokenized_captions["attention_mask"]
        return examples


    # Now you can apply transformations like so
    train_dataset = train_dataset.with_transform(preprocess_train)

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = train_dataset.with_transform(preprocess_train)

    # assert stop

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        gainmap_values = torch.stack([example["gainmap_values"] for example in examples])
        gainmap_values = gainmap_values.to(memory_format=torch.contiguous_format).float()
        
        input_ids = torch.stack([example["input_ids"] for example in examples])
        input_ids = input_ids.to(memory_format=torch.contiguous_format).long()
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        attention_mask = attention_mask.to(memory_format=torch.contiguous_format).float()

        return {
                    "pixel_values": pixel_values, 
                    "gainmap_values": gainmap_values,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    gm_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=gm_optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    gm_unet, gm_optimizer, train_dataloader, gm_lr_scheduler, text_encoder = accelerator.prepare(
        gm_unet, gm_optimizer, train_dataloader, gm_lr_scheduler, text_encoder
    )

    if args.use_ema:
        if args.offload_ema:
            ema_gm_unet.pin_memory()
        else:
            ema_gm_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts", None)
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(gm_unet):
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                text_cond = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

                # debug_dir = os.path.join(args.output_dir, "debug")
                # os.makedirs(debug_dir, exist_ok=True)
                # torchvision.utils.save_image(batch["pixel_values"].cpu(), os.path.join(debug_dir, f"pixel_value.png"))
                # torchvision.utils.save_image(batch["gainmap_values"].cpu(), os.path.join(debug_dir, f"gainmap_value.png"))
                # assert stop

                # Convert images to latent space
                sdr_latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                gm_latents = vae.encode(batch["gainmap_values"].to(weight_dtype)).latent_dist.sample() 
                sdr_latents = sdr_latents * vae.config.scaling_factor
                gm_latents = gm_latents * vae.config.scaling_factor


                gm_noise = torch.randn_like(gm_latents)
                if args.noise_offset:
                    gm_noise += args.noise_offset * torch.randn((gm_noise.shape[0], gm_noise.shape[1], 1, 1), device=gm_noise.device)
                if args.input_perturbation:
                    new_gm_noise = gm_noise + args.input_perturbation * torch.randn_like(gm_noise)
                bsz = gm_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=gm_latents.device)
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_gm_latents = noise_scheduler.add_noise(gm_latents, new_gm_noise, timesteps)
                else:
                    noisy_gm_latents = noise_scheduler.add_noise(gm_latents, gm_noise, timesteps)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    gm_target = gm_noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                input_latents = torch.cat([sdr_latents, noisy_gm_latents], dim=1)
                # Predict the noise residual and compute loss
                gm_model_pred = gm_unet(input_latents, timesteps, text_cond, return_dict=False)[0]
                
                # loss_sdr = F.mse_loss(sdr_model_pred.float(), sdr_target.float(), reduction="mean")
                loss_gm = F.mse_loss(gm_model_pred.float(), gm_target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                # avg_loss_sdr = accelerator.gather(loss_sdr.repeat(args.train_batch_size)).mean()
                avg_loss_gm = accelerator.gather(loss_gm.repeat(args.train_batch_size)).mean()

                # Update the total training loss
                # train_loss += (avg_loss_sdr.item() + avg_loss_gm.item()) / args.gradient_accumulation_steps
                train_loss += avg_loss_gm.item() / args.gradient_accumulation_steps

                accelerator.backward(loss_gm)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(gm_unet.parameters(), args.max_grad_norm)
                gm_optimizer.step()
                gm_lr_scheduler.step()  # If you have a separate scheduler for gm_model
                gm_optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    if args.offload_ema:
                        ema_gm_unet.to(device="cuda", non_blocking=True)
                    ema_gm_unet.step(gm_unet.parameters())
                    if args.offload_ema:
                        ema_gm_unet.to(device="cpu", non_blocking=True)

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"avg_loss_gm": avg_loss_gm}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_gm_unet.copy_to(gm_unet.parameters())
                                        
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            gm_unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_gm_unet.restore(gm_unet.parameters())

            logs = {"gm_loss": loss_gm.detach().item(),
                    "lr": gm_lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        gm_unet = unwrap_model(gm_unet)
        if args.use_ema:
            ema_gm_unet.copy_to(gm_unet.parameters())

        gm_pipeline = StableDiffusionGMPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(gm_unet),
            safety_checker=None,
            scheduler=noise_scheduler,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        gm_pipeline.save_pretrained(os.path.join(args.output_dir, "save_pipeline"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
