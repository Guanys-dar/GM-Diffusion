#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os
import torch
from PIL import Image
from torchvision import transforms
# from gm_diffusion.pipelines import StableDiffusionGMPipeline
from gm_diffusion.pipelines import StableDiffusionDualUNetPipeline
# from gm_diffusion.pipelines_improve import StableDiffusionDualUNetPipeline
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
# Change to DPMSolver++ for better quality
from diffusers import DPMSolverMultistepScheduler

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
        "/path/to/bVAE"
    )
        
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )    
    sd15_path = r"/path/to/sd15"
    sd_unet = UNet2DConditionModel.from_pretrained(sd15_path, subfolder="unet")

    gm_unet = load_unet_model(args)

    # 加载训练好的模型
    pipeline = StableDiffusionDualUNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=sd_unet,
        gm_unet=gm_unet,
        scheduler=noise_scheduler,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)

    prompts = [
        "Close-up portrait of an elderly woman with deep wrinkles and kind eyes, laughing heartily, natural window lighting casting soft shadows, detailed skin texture, shallow depth of field, 50mm lens.", 
        # "Dynamic action shot of a basketball player mid-air, dunking the ball, dramatic stadium lighting, motion blur on limbs, sweat visible, detailed jersey fabric texture, wide-angle perspective from below.", 
        # "Extreme close-up macro shot of a praying mantis head, compound eyes reflecting the environment, intricate alien-like facial structures, vibrant green exoskeleton, razor-sharp focus", 
        # "Whimsical illustration of a fox wearing a tiny waistcoat and top hat, reading a book under a mushroom, storybook style, soft watercolor textures, warm, gentle lighting.", 
        # "Cutaway technical illustration of a high-performance jet engine, showing intricate turbine blades and combustion chambers, blueprint style lines with color highlights, clean vector look.", 
        # "Stack of antique leather-bound books on a dusty wooden table, gold foil lettering on spines slightly worn, texture of aged paper, dramatic side lighting emphasizing dust particles in the air.", 
        # "Geometric pattern tessellation inspired by M.C. Escher, transforming birds into fish, monochrome, intricate line work, optical illusion.", 
        # "A surreal dreamscape where clocks are melting over floating rocks, Dali-inspired, vast desert background under a purple sky with two moons, painterly but realistic textures", 
        # "A glass sphere sitting on a checkered floor, reflecting the room distortedly, including a bright window and a person standing just out of frame, realistic refraction and reflection.", 
        # "Chainmail armor texture, close-up showing interlocking metal rings, subtle variations in reflection and shadow on each ring, metallic sheen.", 
        # "A crowded medieval marketplace scene with dozens of distinct figures in varied clothing, stalls selling pottery and textiles, detailed background architecture, bright daylight."
        # "Abstract swirling explosion of colorful powder paint (like Holi festival), frozen mid-air, high-speed photography effect, black background, vibrant neon pinks, yellows, blues.",
        # "Vector art logo design for a coffee shop named The Quantum Bean, incorporating an atom symbol and a coffee bean, clean lines, flat colors, minimalist style.",
        # "Sheet music with intricate notes and symbols laid out on a wooden piano, focus sharp on the notes, shallow depth of field blurring the piano keys."
        # "Character concept art for a sci-fi rogue: cybernetic arm, leather jacket with intricate stitching, neon city lights reflecting in sunglasses, gritty atmosphere, detailed facial expression, digital painting style.",
        # "A group of diverse friends sitting around a campfire at night, telling stories, faces illuminated by firelight, starry sky above, long exposure capturing faint star trails, warm and cozy atmosphere.",
        # "Historical portrait painting in the style of Rembrandt: a scholar examining an ancient globe, strong chiaroscuro lighting, rich dark tones, visible brushstrokes, texture of aged paper and wood.",
        # "Street style photograph of a person walking through a rainy Tokyo intersection at night, reflections on wet pavement, neon signs blurring in the background, transparent umbrella catching raindrops, sharp focus on the subject.",
    ]

    quality_suffix = "HDR10, ultra-sharp details, professional color grading, no artifacts"
    prompts = [quality_suffix + p for p in prompts]

    negative_prompt = (
    "worst quality, normal quality, low quality, low res, blurry, distortion, text, watermark, "
    "logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, "
    "duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, "
    "bad proportions, deformed, disconnected limbs, out of frame, out of focus, dehydrated, "
    "disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, "
    "malformed limbs, mutated, mutated hands, missing arms, missing fingers, poorly drawn hands, "
    "poorly drawn face, pixelated, grainy, color aberration, amputee, bad illustration, "
    "blank background, body out of frame, boring background, cut off, dismembered, distorted, "
    "draft, extra fingers, extra legs, hazy, low resolution, malformed, missing hands, "
    "missing legs, mutilated, unattractive, unnatural pose, unreal engine"
    )


    seed = 1231
    generator = torch.Generator(device=device).manual_seed(seed)

    count = 0
    for prompt in prompts:
        # 模型预测
        with torch.no_grad():
            count += 1
            sdr_input_filename = f"dual_unet_{count}.png"
        
            sdr_latent, gm_latent = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=80,  # Increased from 50
                guidance_scale=8.0,  # Stronger prompt adherence
                noise_level=0.0,
                generator=generator,
                output_type="latent",
                eta=0.7,  # Controls stochasticity
                cross_attention_kwargs={"scale": 0.8},  # For better text alignment
                use_karras_sigmas=True
            )

            # 解码 GM 图像
            sdr_latent = 1 / vae.config.scaling_factor * sdr_latent
            sdr_image_decoded = pipeline.vae.decode(sdr_latent, return_dict=False)[0]
            sdr_image_decoded = (sdr_image_decoded / 2 + 0.5).clamp(0, 1)
            sdr_image_decoded = sdr_image_decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            

            gm_latent = 1 / vae.config.scaling_factor * gm_latent
            gm_image = pipeline.vae.decode(gm_latent, return_dict=False)[0]
            gm_image = (gm_image / 2 + 0.5).clamp(0, 1)
            gm_image = gm_image.cpu().permute(0, 2, 3, 1).float().numpy()[0]

        # 保存 SDR 和 GM 图像
        sdr_image_path = os.path.join(args.output_dir, f"sdr_{sdr_input_filename}")
        gm_image_path = os.path.join(args.output_dir, f"gm_{sdr_input_filename}")
        Image.fromarray((sdr_image_decoded * 255).astype(np.uint8)).save(sdr_image_path)
        Image.fromarray((gm_image * 255).astype(np.uint8)).save(gm_image_path)

        qmax = 99
        hdr_image = apply_gm_to_sdr(
            sdr=sdr_image_decoded,
            gm=gm_image,
            qmax=qmax)

        print(f"hdr_image range:",hdr_image.max(),hdr_image.min())

        save_hdr_filename = sdr_input_filename.replace("png","hdr")

        save_hdr_image(
            hdr_image, 
            args.output_dir, 
            f"hdr_{save_hdr_filename}", 
            qmax)

if __name__ == "__main__":
    main()