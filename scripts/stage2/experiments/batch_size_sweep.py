#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os
import torch
from PIL import Image
from torchvision import transforms
from gm_diffusion.pipelines import StableDiffusionGMPipeline
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 设置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
import cv2

def save_hdr_image(apply_HDR, save_dir, filename, qmax):
    apply_HDR = apply_HDR / (qmax+1)
    apply_HDR_image = apply_HDR.astype(np.float32)
    cv2.imwrite(os.path.join(save_dir, f"{filename}"), apply_HDR_image[:,:,[2,1,0]])

def apply_gm_to_sdr(gm, sdr, qmax=9, eps=1/64):
    """
    Applies the given GM (Guidance Map) to SDR (Standard Dynamic Range) image to compute HDR (High Dynamic Range) output.
    Note: SDR and GM should be NumPy arrays with values in [0, 1] for SDR.
    """
    # Ensure SDR is within [0, 1] and linearize using sRGB gamma
    sdr_clamped = np.clip(sdr, 0, 1)
    sdr_linear = sdr_clamped ** 2.2

    # Compute HDR output
    output_hdr = (sdr_linear + eps) * (1 + gm * qmax) - eps
    return output_hdr

def parse_args():
    parser = argparse.ArgumentParser(description="Test the trained model.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to the trained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_ckpt",
        type=str,
        default=None,
        required=True,
        help="Path to the trained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--sdr_input_path",
        type=str,
        default=None,
        required=True,
        help="Path to the input SDR image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_outputs",
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible testing.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for validation images.",
    )
    return parser.parse_args()

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

def load_unet_model(args):
    # Load the original configuration from the checkpoint
    
    gm_unet_config = UNet2DConditionModel.load_config(args.unet_ckpt)
    gm_unet_config["_name_or_path"] = str(args.unet_ckpt)

    if "num_attention_heads" in gm_unet_config:
        attention_head_dim = gm_unet_config.pop("num_attention_heads")
        gm_unet_config["attention_head_dim"] = attention_head_dim
        
        unet_ckpt_path = Path(args.unet_ckpt)
        config_path = unet_ckpt_path / "config.json"
        
        unet_ckpt_path.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(gm_unet_config, f, indent=4)
            print(f"Configuration updated and saved to {config_path}")

    # Define the configuration parameters
    config = {
        "_class_name": "UNet2DConditionModel",
        "_diffusers_version": "0.6.0",
        "act_fn": "silu",
        "attention_head_dim": 8,
        "block_out_channels": [320, 640, 1280, 1280],
        "center_input_sample": False,
        "cross_attention_dim": 768,
        "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
        "downsample_padding": 1,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "layers_per_block": 2,
        "mid_block_scale_factor": 1,
        "norm_eps": 1e-05,
        "norm_num_groups": 32,
        "out_channels": 4,
        "sample_size": 64,
        "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"]
    }

    # Load the model with the specified configuration
    gm_unet = UNet2DConditionModel.from_pretrained(
        args.unet_ckpt,
        in_channels=8,
        **config  # Unpack the configuration dictionary
    )
    return gm_unet

def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型组件
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
        
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )    

    gm_unet = load_unet_model(args)

    # 加载训练好的模型
    pipeline = StableDiffusionGMPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=gm_unet,
        scheduler=noise_scheduler,
    )

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # half precision enable 3 minutes original resolution inference.
    pipeline = pipeline.to(device, dtype=torch.float16)
    
    # add half
    # pipeline.unet = pipeline.unet.half()
    # pipeline.vae = pipeline.vae.half()
    # pipeline.text_encoder = pipeline.text_encoder.half()

    # 加载输入图像
    # res_width = int(args.resolution / 16) * 9
    # res_height = args.resolution
    # print(res_width,res_height)

    val_transforms = transforms.Compose(
        [
            # transforms.Resize((res_width,res_height), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.ConvertImageDtype(torch.float16)  # Half precision
        ]
    )


    # try batch inference
    sdr_input_paths = sorted(Path(args.sdr_input_path).rglob("*.png"))
    sdr_input_paths = sdr_input_paths[-5:-1]

    for batch_idx in tqdm(range(0, len(sdr_input_paths), args.batch_size), desc="Processing batches"):
        batch_paths = sdr_input_paths[batch_idx:batch_idx+args.batch_size]
        batch_images = []
        original_images = []
        filenames = []
        sdr_images_decoded_list = {}
        
        # Load and transform batch
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            original_images.append(img.copy())
            batch_images.append(val_transforms(img).unsqueeze(0))
            filenames.append(os.path.basename(path))
            
        batch_tensor = torch.cat(batch_images, dim=0).to(device)

        # Batch inference
        with torch.no_grad():
            sdr_latent = pipeline.vae.encode(batch_tensor).latent_dist.sample()
            sdr_latent = sdr_latent * pipeline.vae.config.scaling_factor

            # Process entire batch
            gm_latent = pipeline(
                sdr_latent,
                prompt=[""] * len(batch_paths),  # Batch-sized prompt list
                num_inference_steps=50,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                output_type="latent",
            ).images

            # Decode batch
            sdr_latent_decoded = 1 / vae.config.scaling_factor * sdr_latent
            sdr_images_decoded = pipeline.vae.decode(sdr_latent_decoded, return_dict=False)[0]
            sdr_images_decoded = (sdr_images_decoded / 2 + 0.5).clamp(0, 1)
            sdr_images_decoded = sdr_images_decoded.cpu().permute(0, 2, 3, 1).float().numpy()

            gm_latent_decoded = 1 / vae.config.scaling_factor * gm_latent
            gm_images = pipeline.vae.decode(gm_latent_decoded, return_dict=False)[0]
            gm_images = (gm_images / 2 + 0.5).clamp(0, 1)
            gm_images = gm_images.cpu().permute(0, 2, 3, 1).float().numpy()
            
            sdr_images_decoded_list[f"{idx}"] = sdr_images_decoded

        # Process individual results
        for idx, (orig_img, filename) in enumerate(zip(original_images, filenames)):
            # Resize GM to original image size
            gm_pil = Image.fromarray((gm_images[idx] * 255).astype(np.uint8))
            gm_resized = gm_pil.resize(orig_img.size, Image.BILINEAR)
            gm_final = np.array(gm_resized).astype(np.float32) / 255

            # Save outputs
            Image.fromarray((sdr_images_decoded[idx] * 255).astype(np.uint8)).save(
                os.path.join(args.output_dir, f"sdr_{filename}")
            )
            Image.fromarray((gm_images[idx] * 255).astype(np.uint8)).save(
                os.path.join(args.output_dir, f"gm_{filename}")
            )

            # Generate HDR
            orig_sdr = np.array(orig_img).astype(np.float32) / 255
            hdr_image = apply_gm_to_sdr(orig_sdr, gm_final, qmax=99)
            save_hdr_filename = filename.replace("png", "hdr")
            save_hdr_image(hdr_image, args.output_dir, f"orginal_hdr_{save_hdr_filename}", 99)

            sdr_decode = sdr_images_decoded_list[f"{idx}"]
            hdr_image = apply_gm_to_sdr(orig_sdr, gm_final, qmax=99)
            save_hdr_filename = filename.replace("png", "hdr")
            save_hdr_image(hdr_image, args.output_dir, f"batch_hdr_{save_hdr_filename}", 99)

if __name__ == "__main__":
    main()