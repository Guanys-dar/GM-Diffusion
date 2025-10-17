#!/usr/bin/env python
# coding=utf-8
"""Stage 3: Generate paired SDR and HDR (via GM) images using the fine-tuned model."""

import argparse
import logging
import os
import torch
from PIL import Image
from torchvision import transforms
from gm_diffusion.pipelines import StableDiffusionGMPipeline
from gm_diffusion.stage1 import apply_gm_to_sdr
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
    pipeline = pipeline.to(device)

    # 加载输入图像
    # res_width = int(args.resolution / 16) * 9
    # res_height = args.resolution
    # print(res_width,res_height)

    val_transforms = transforms.Compose(
        [
            # transforms.Resize((res_width,res_height), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    sdr_input_paths = sorted(Path(args.sdr_input_path).rglob("*.png"))
    for sdr_input_path in tqdm(sdr_input_paths, desc="Processing SDR images"):
        sdr_input_filename = os.path.basename(sdr_input_path)
        sdr_image = Image.open(sdr_input_path).convert("RGB")
        original_sdr_image = sdr_image.copy()
        print("before resize:", original_sdr_image.size)
        sdr_image = val_transforms(sdr_image).unsqueeze(0).to(device)
        print("after resize, image:", sdr_image.shape)

        # 模型预测
        with torch.no_grad():
            # 编码 SDR 图像
            sdr_latent = pipeline.vae.encode(sdr_image).latent_dist.sample()
            sdr_latent = sdr_latent * pipeline.vae.config.scaling_factor

            # 生成 Gain Map (GM)
            gm_latent = pipeline(
                sdr_latent,
                prompt=[""],
                num_inference_steps=50,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                output_type="latent",
            ).images[0]
            gm_latent = gm_latent.unsqueeze(0)

            # print("sdr_latent.shape", sdr_latent.shape)
            # print("gm_latent.shape", gm_latent.shape)

            # 解码 GM 图像
            sdr_latent = 1 / vae.config.scaling_factor * sdr_latent
            sdr_image_decoded = pipeline.vae.decode(sdr_latent, return_dict=False)[0]
            sdr_image_decoded = (sdr_image_decoded / 2 + 0.5).clamp(0, 1)
            sdr_image_decoded = sdr_image_decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]

            gm_latent = 1 / vae.config.scaling_factor * gm_latent
            gm_image = pipeline.vae.decode(gm_latent, return_dict=False)[0]
            gm_image = (gm_image / 2 + 0.5).clamp(0, 1)
            gm_image = gm_image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            print("gm_image.shape", gm_image.shape)

        # # resize the gm_image into the same size as the original_sdr_image
        # gm_image_resized = Imagze.fromarray((gm_image * 255).astype(np.uint8)).resize(original_sdr_image.size, Image.BILINEAR)
        # gm_image = cv2.resize(gm_image, (original_sdr_image.size[0], original_sdr_image.size[1]), interpolation=cv2.INTER_LINEAR)
        # gm_image = np.array(gm_image_resized).astype(np.float32) / 255

        # 保存 SDR 和 GM 图像
        sdr_image_path = os.path.join(args.output_dir, f"sdr_{sdr_input_filename}")
        gm_image_path = os.path.join(args.output_dir, f"gm_{sdr_input_filename}")
        Image.fromarray((sdr_image_decoded * 255).astype(np.uint8)).save(sdr_image_path)
        Image.fromarray((gm_image * 255).astype(np.uint8)).save(gm_image_path)

        # logger.info(f"SDR image saved to {sdr_image_path}")
        # logger.info(f"GM image saved to {gm_image_path}")

        sdr_image = (sdr_image / 2 + 0.5).clamp(0, 1)
        sdr_image = sdr_image.cpu().permute(0, 2, 3, 1).float().numpy()[0]

        # sdr_image_decoded = cv2.resize(sdr_image_decoded, (original_sdr_image.size[0], original_sdr_image.size[1]), interpolation=cv2.INTER_LINEAR)
        # gm_image = cv2.resize(gm_image, (original_sdr_image.size[0], original_sdr_image.size[1]), interpolation=cv2.INTER_LINEAR)

        qmax = 99
        hdr_image = apply_gm_to_sdr(
            sdr=sdr_image_decoded,
            gm=gm_image,
            qmax=qmax)

        original_hdr_image = apply_gm_to_sdr(
            sdr=original_sdr_image,
            gm=gm_image,
            qmax=qmax)


        print(f"hdr_image range:",hdr_image.max(),hdr_image.min())

        save_hdr_filename = sdr_input_filename.replace("png","hdr")

        save_hdr_image(
            hdr_image, 
            args.output_dir, 
            f"hdr_{save_hdr_filename}", 
            qmax)

        save_hdr_image(
            original_hdr_image, 
            args.output_dir, 
            f"original_hdr_{save_hdr_filename}", 
            qmax)

if __name__ == "__main__":
    main()
