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
        args.pretrained_model_name_or_path, subfolder="vae"
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
        "A stunning 8K HDR landscape with crystal-clear sky, majestic snow-capped mountains, intricate cloud details, ultra-sharp foreground, cinematic lighting, unreal engine 5 rendering",
        "Lush vibrant green field with photorealistic vegetation, hyperdetailed ancient oak trees, rolling hills in distance, golden hour lighting, depth of field, award-winning nature photography",
        "Impressionist oil painting of river through autumn forest: vibrant orange foliage, thick brushstrokes, visible canvas texture, Claude Monet style, gallery-quality artwork",
        "Artisan woven baskets with cobalt blue accents, intricate wicker patterns, natural fiber textures, studio lighting, product photography, 4k resolution",        # Use reference quality indicators
        "Ultra-detailed artist's color palette: visible oil paint blobs, metallic pigment reflections, wooden texture background, macro photography, f/2.8 aperture",
        "Dramatic sunset over mirror-calm lake: intense orange-purple gradient sky, silhouette pine trees foreground, HDR 16-bit color depth, peak luminance 1000 nits",
        "Macro shot of dewdrops clinging to spider silk at dawn: crystalline refraction of sunrise colors, razor-thin focus plane, visible surface tension physics, 100mm macro lens, f/5.6 aperture, ultra-high ISO clarity in shadow details.",
        "Surreal floating islands with waterfalls cascading into clouds: moss-covered rock formations, bioluminescent flora glowing amber, volumetric fog layers, sunset backlighting, digital matte painting style with photorealistic texture mapping.",
        "Neon-drenched cyberpunk alleyway: holographic store signs flickering in rain, neon pink/purple/teal color grading, hyperdetailed wet pavement reflections, cinematic wide-angle distortion, Blade Runner 2049 aesthetic with RTX ray tracing effects.",
        "Abandoned Art Deco theater interior: peeling gold leaf columns, velvet seats consumed by ivy, sunlight piercing through collapsed roof, medium format film grain, chiaroscuro lighting emphasizing dust motes in light beams.",
        "Slow-motion capture of ink dispersing in liquid mercury: iridescent metallic tendrils swirling in zero gravity, 1000fps temporal detail, dark void background, laboratory lighting with specular highlights on fluid surfaces.",
        "Erupting volcano under aurora borealis: incandescent lava rivers illuminating ash clouds, green celestial ribbons reflected in obsidian-black obsidian fields, thermal camera color palette blended with astronomical photography.",
        "Microscopic view of oxidized copper patina: turquoise/green crystal growth patterns, electron microscope-level detail, metallic subsurface scattering, textured depth map lighting, 8K procedural material study.",
        "Kaleidoscopic mandala of autumn leaves: fractal symmetry with maple/orange/gold hues, optical prism effects creating spectral flares, 3D rendered with subsurface scattering on translucent leaf veins, 4D noise displacement.",
        "Ancient library swallowed by jungle: stone arches draped with orchids, sunlight filtering through vine-choked skylights, leather-bound tomes sprouting mushrooms, cinematic fog atmosphere, Unreal Engine 5 nanite geometry.",
        "Steampunk airship docking at cloud city: brass machinery with visible gear mechanisms, hydrogen gasbags glowing from internal lanterns, Victorian-era costumes with fabric physics, dusk lighting with gaslamp warmth against cool skyscraper shadows.",
        "Tilt-shift miniature effect on Tokyo nightscape: selective focus on glowing train lines, bokeh streetlights as golden orbs, hyperreal scaling illusion, f/3.5 aperture simulation, diorama-style post-processing.",
        "Underwater cathedral ruins: sunrays piercing turquoise depths, stone arches covered in coral formations, schooling fish creating dynamic light patterns, medium format Hasselblad color science with anti-water-distortion algorithms.",
    ]

    test_prompts = [
        "A serene alpine landscape at golden hour, featuring a crystal-clear lake mirroring snow-capped mountains, surrounded by dense evergreen forests with morning mist rising between the trees. Ultra-detailed 4K resolution, vibrant emerald greens and azure waters, cinematic lighting with sun rays piercing through clouds, realistic texture details in pine needles and rocky peaks.",
        "Dramatic close-up of intense orange flames with blue core temperatures consuming dry logs in pitch-black darkness, glowing embers floating upward, smoke wisps curling through the air. Hyper-realistic fire texture with heat distortion effects, 8K resolution, chiaroscuro lighting emphasizing the contrast between fiery warmth and cold void surroundings.",
        "Wide-angle view of vibrant fireworks exploding in a gradient twilight sky over a metropolitan skyline, capturing multiple burst stages with trailing golden sparks and neon-colored starburst patterns. Reflections shimmering on a river below, ultra-high detail in smoke plumes and light trails, rich saturation in magenta, cyan and amber hues, realistic long-exposure photography style.",
        "Dramatic moment of a jagged purple-white lightning bolt splitting a turbulent indigo sky above a windswept prairie, instantaneous illumination revealing swirling storm clouds and bent grasses. High-speed photography detail with visible branching patterns, realistic atmospheric perspective, raindrops frozen mid-air by the flash, HDR-enhanced contrast.",
        "A rustic wooden window frame with fluttering linen curtains revealing a hyper-realistic garden panorama in midday sun - lush emerald meadows dotted with wildflowers, distant birch forest, and cotton-cloud skies. Shallow depth of field emphasizing window texture details while maintaining crisp background clarity, natural color grading with vibrant chromatic contrast between interior shadows and exterior brilliance.",
    ]
    prompts = test_prompts + prompts

    # Add consistent quality suffixes
    quality_suffix = "HDR10, ultra-sharp details, professional color grading, no artifacts"
    prompts = [ p+quality_suffix for p in prompts]
    # negative_prompt = "low quality, blurry, distorted, underexposed, overexposed, lowres, artifacts"
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


    seed = 4369
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
                num_inference_steps=75,  # Increased from 50
                guidance_scale=9.0,  # Stronger prompt adherence
                noise_level=0.0,
                generator=generator,
                output_type="latent",
                eta=0.7,  # Controls stochasticity
                cross_attention_kwargs={"scale": 0.8},  # For better text alignment
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