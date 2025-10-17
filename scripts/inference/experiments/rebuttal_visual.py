#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os
import torch
import math # Added math
from PIL import Image, ImageDraw, ImageFont # Added ImageDraw, ImageFont
from torchvision import transforms
# from gm_diffusion.pipelines import StableDiffusionGMPipeline
# from gm_diffusion.pipelines import StableDiffusionDualUNetPipeline
from visualize_latents import StableDiffusionDualUNetPipelineVis
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

# NEW Helper Function: Decode latents to PIL images
def decode_latents_to_pil(vae, latents_list):
    images = []
    if not latents_list:
        return images

    # Move latents back to GPU for decoding if needed
    device = vae.device
    latents_dtype = vae.dtype

    for latents in tqdm(latents_list, desc="Decoding intermediate steps"):
        latents = latents.to(device, dtype=latents_dtype) # Move back to GPU
        latents = 1 / vae.config.scaling_factor * latents
        image = vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # We decode one by one, so select the first image from the batch
        image = image.cpu().detach().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).astype(np.uint8)
        images.append(Image.fromarray(image))
    return images

# NEW Helper Function: Create Visualization Grid
def create_visualization_grid(sdr_images, gm_images, output_path, steps_to_show=10):
    if not sdr_images or not gm_images:
        print("No intermediate images to visualize.")
        return

    # Select a subset of steps for visualization if too many
    num_steps = len(sdr_images)
    if num_steps > steps_to_show:
        indices = np.linspace(0, num_steps - 1, steps_to_show, dtype=int)
        print(f"Vis Steps: {indices}")
        sdr_images = [sdr_images[i] for i in indices]
        gm_images = [gm_images[i] for i in indices]
    else:
        indices = list(range(num_steps)) # Keep track of original step index if needed

    n_images = len(sdr_images)
    if n_images == 0:
        return

    img_w, img_h = sdr_images[0].size
    grid_w = img_w * n_images
    grid_h = img_h * 2 # Two rows

    grid_image = Image.new('RGB', (grid_w, grid_h))

    for i, (sdr_img, gm_img) in enumerate(zip(sdr_images, gm_images)):
        grid_image.paste(sdr_img, (i * img_w, 0))       # Top row: SDR
        grid_image.paste(gm_img,  (i * img_w, img_h))  # Bottom row: GM

        # Optional: Add step number label (requires font)
        # try:
        #     draw = ImageDraw.Draw(grid_image)
        #     font = ImageFont.truetype("arial.ttf", 15) # Adjust font path/size
        #     step_label = f"Step {indices[i]}"
        #     # Position labels for SDR row
        #     draw.text((i * img_w + 5, 5), step_label, fill="white", font=font)
        #     # Position labels for GM row
        #     draw.text((i * img_w + 5, img_h + 5), step_label, fill="white", font=font)
        # except IOError:
        #     print("Font file not found, skipping labels.")


    grid_image.save(output_path)
    print(f"Saved visualization grid to {output_path}")

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
    try:
        pipeline = StableDiffusionDualUNetPipelineVis.from_pretrained( # Use the subclass
            args.pretrained_model_name_or_path, # Base model path for non-UNet components
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=sd_unet,     # Pass the loaded base SD UNet
            gm_unet=gm_unet,  # Pass the loaded GM UNet
            scheduler=noise_scheduler,
            # Optional: Disable safety checker if not needed/available
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        # Set the scheduler (DPMSolver) after loading
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        logger.info(f"Pipeline loaded successfully. Scheduler: {pipeline.scheduler.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return # Exit if pipeline fails to load


    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    try:
        pipeline = pipeline.to(device)
        vae = vae.to(device) # Also move VAE to device for decoding
    except Exception as e:
        logger.error(f"Failed to move models to device {device}: {e}")
        return

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
    quality_suffix = ", 8k resolution, HDR10, ultra-sharp details, professional color grading, no artifacts"
    prompts = [p + quality_suffix for p in prompts]
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


    base_seed = 4369
    generator = torch.Generator(device=device).manual_seed(base_seed)

    count = 0
    for prompt in tqdm(prompts, desc="Generating images"):
        count += 1
        base_filename = f"dual_unet_{count:03d}" # Use formatted count for sorting
        # Use a consistent generator for each prompt run for reproducibility if desired
        # Or vary the seed slightly per prompt
        current_seed = base_seed # + count # Or just base_seed
        generator = torch.Generator(device=device).manual_seed(current_seed)
        logger.info(f"\n--- Processing prompt {count}/{len(prompts)} (Seed: {current_seed}) ---")
        logger.info(f"Prompt: {prompt[:100]}...") # Log truncated prompt

        # === Call the pipeline with visualization arguments ===
    
        with torch.no_grad():
            pipeline_output = pipeline( # Call the subclass instance
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=75,
                guidance_scale=9.0,
                generator=generator,
                # === Request intermediates ===
                output_type="pil",           # Request PIL images directly
                return_dict=True,            # Required
                return_intermediates=True,     # Enable intermediate capture
                # === Other parameters ===
                eta=0.7, # Keep if used by DPMSolver
                cross_attention_kwargs={"scale": 0.8}, # Keep if needed
                # Removed noise_level=0.0 as it wasn't in the standard pipeline call
            )

        # === Extract results from the dictionary ===
        if "sdr_image" not in pipeline_output or "gm_image" not in pipeline_output:
            logger.error(f"Pipeline output dictionary missing 'sdr_image' or 'gm_image' for prompt {count}. Skipping.")
            continue

        # Final images (should be PIL)
        sdr_image_final_pil = pipeline_output["sdr_image"][0] # Output is a list
        gm_image_final_pil = pipeline_output["gm_image"][0]

        # Intermediates (should be lists of latents)
        intermediate_sdr_latents = pipeline_output.get("sdr_intermediates", [])
        intermediate_gm_latents = pipeline_output.get("gm_intermediates", [])
        logger.info(f"Captured {len(intermediate_sdr_latents)} intermediate steps.")

        # --- Decode Intermediate Images ---
        intermediate_sdr_images = decode_latents_to_pil(vae, intermediate_sdr_latents)
        intermediate_gm_images = decode_latents_to_pil(vae, intermediate_gm_latents)

        # --- Prepare Numpy versions for saving/HDR ---
        # Convert PIL [0,255] to Numpy [0,1] float
        sdr_image_final_np = np.array(sdr_image_final_pil).astype(np.float32) / 255.0
        gm_image_final_np = np.array(gm_image_final_pil).astype(np.float32) / 255.0

        # --- Save Final Outputs ---
        output_dir = "/path/to/rebuttal_outputs/"
        sdr_image_path =  f"{output_dir}/sdr_{base_filename}.png"
        gm_image_path = f"{output_dir}/gm_{base_filename}.png"
        hdr_image_path = f"{output_dir}/hdr_{base_filename}.hdr" # Keep .hdr extension
        vis_grid_path = f"{output_dir}/vis_{base_filename}.png" # Path for the grid

        logger.info(f"Saving final SDR image to {sdr_image_path}")
        sdr_image_final_pil.save(sdr_image_path)
        logger.info(f"Saving final GM image to {gm_image_path}")
        gm_image_final_pil.save(gm_image_path)

        # Apply GM and save HDR (using numpy arrays in [0,1] range)
        qmax = 99
        logger.info(f"Applying GM to SDR (qmax={qmax}) to generate HDR...")
        hdr_image = apply_gm_to_sdr(
            sdr=sdr_image_final_np,
            gm=gm_image_final_np, # Use GM numpy array
            qmax=qmax
        )
        # Clamp negative values potentially introduced by the formula
        hdr_image_clipped = np.clip(hdr_image, 0, None)
        max_val = hdr_image_clipped.max()
        min_val = hdr_image_clipped.min()
        logger.info(f"Generated HDR image range (after clip>=0): Min={min_val:.4f}, Max={max_val:.4f}")
        logger.info(f"Saving final HDR image to {hdr_image_path}")
        # Pass the clipped HDR image to save function
        save_hdr_image(hdr_image_clipped, output_dir, f"hdr_{base_filename}.hdr", qmax)

        # --- Create and Save Visualization Grid ---
        logger.info(f"Creating visualization grid...")
        create_visualization_grid(
            intermediate_sdr_images,
            intermediate_gm_images,
            vis_grid_path,
            steps_to_show=8# Use argument for number of steps
        )

    logger.info("--- Processing finished ---")

if __name__ == "__main__":
    main()
