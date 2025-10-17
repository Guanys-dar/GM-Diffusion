# HDR Image Generation via Gain Map Decomposed Diffusion

Offical Repo for ICCV 2025 — “HDR Image Generation via Gain Map Decomposed Diffusion”   
Confidential review copy. Do not distribute.

Our method tackles two long-standing obstacles for HDR image generation:
pretrained SDR auto-encoders clip 16-bit HDR signals, and large-scale HDR datasets are scarce.
We resolve both by adopting the industry “double-layer” HDR representation—each HDR frame is
factored into an SDR base layer plus a low-bit-depth gain map (GM)—and by constructing
unsupervised Text–SDR–GM triples from commodity datasets such as MSCOCO.

The full system follows the three-stage recipe described in the paper:

1. **Stage 1 – Unsupervised Text–SDR–GM construction**  
   Adapt the Stable Diffusion VAE with LoRA while applying brightness-aware compression and
   gamut-constrained reduction. The tuned VAE predicts gain maps that, together with the SDR
   inputs, recover HDR outputs via Eq. (1) in the paper.
2. **Stage 2 – GM diffusion fine-tuning**  
   Condition a Stable Diffusion UNet on SDR latents and jointly denoise SDR and GM latents so
   that the model learns pixel-aligned gain-map generation across diffusion steps.
3. **Stage 3 – Decomposed HDR inference**  
   Run the dual-branch pipeline to jointly synthesise SDR renders and their gain maps, then
   reconstruct HDR frames (e.g., for text-to-HDR synthesis, ControlNet-driven generation, and
   SDR-to-HDRTV up-conversion).

## Repository Layout

```
gm_diffusion/           # Core Python package
├── stage1/             # Augmentations, tone-mapping ops, discriminator
└── pipelines/          # Stable Diffusion pipelines (single & dual UNet variants)
scripts/
├── stage1/             # Stage 1 LoRA fine-tuning CLI
├── stage2/             # Stage 2 GM diffusion training CLI (+ experiments)
└── inference/          # Stage 3 inference CLI (+ experiments)
```

Experimental scripts used for ablations and rebuttal figures remain under
`scripts/*/experiments/`; they contain placeholder paths (`/path/to/...`) you should override
when reproducing specific figures.

## Getting Started

### Environment

Target Python 3.10+ with CUDA 11.8 or newer. Example bootstrap:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers>=0.33 transformers accelerate datasets peft timm wandb opencv-python
```

Install optional accelerators (`xformers`, `bitsandbytes`) as needed and run `accelerate config`
before multi-GPU launches.

### Data Prerequisites

Stages 1 and 2 consume parquet manifests that describe the automatically generated triple
dataset. Each row should minimally store:

| column     | description                                               |
|------------|-----------------------------------------------------------|
| `image`    | MSCOCO SDR image bytes or file path                       |
| `text`     | paired caption/prompt                                     |
| `gainmap`  | encoded gain map (byte array or path)                     |
| `qmax`     | maximum gain value used in Eq. (1) (optional but useful)  |

Use `--train_metadata` to point the training scripts at one or many parquet files. Caching is
handled through Hugging Face `datasets` with `--dataset_cache_dir`.

## Stage 1 – LoRA-Tuned VAE for Gain Map Prediction

We fine-tune the Stable Diffusion VAE with LoRA so it learns to reconstruct SDR/Gain-map pairs
while respecting the brightness-aware compression and gamut constraints described in Sec. 3.2 of
the paper.

```bash
accelerate launch scripts/stage1/train_vqgan_lora.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_metadata data/stage1/train_metadata.parquet \
  --dataset_cache_dir data/cache \
  --bright_tmo fix_mulog_tmo \
  --train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --output_dir outputs/stage1_vqgan
```

Notable flags:

- `--bright_tmo`: choose one of `{hard_clip_tmo, linear_scale_tmo, fix_mulog_tmo}` to match the
  paper’s tone-mapping operators.
- `--clip_pixel`: enable the RandomExposureAdjust augmentation that approximates camera response
  variation.

## Stage 2 – Gain Map Diffusion Training

Stage 2 adapts the Stable Diffusion UNet to predict gain maps conditioned on SDR latents,
enforcing pixel-level alignment as in Sec. 3.3.

```bash
accelerate launch scripts/stage2/train_gm_unet.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_metadata data/stage2/*.parquet \
  --dataset_cache_dir data/cache \
  --validation_prompt_file configs/validation_prompts.txt \
  --validation_image_dir data/validation_sdr \
  --scheduler_config configs/pndm_scheduler \
  --output_dir outputs/stage2_gm_unet
```

Tip: Provide either `--validation_prompts` on the CLI or a `--validation_prompt_file`, and pair
them with SDR reference images via `--validation_image_dir` to reproduce the validation figures.

## Stage 3 – Text-to-HDR and SDR-to-HDRTV Inference

The inference driver reconstructs HDR outputs by combining predicted SDR renders and gain maps
according to Eq. (1).

```bash
python scripts/inference/generate_hdr.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --unet_ckpt outputs/stage2_gm_unet/checkpoint-123456 \
  --sdr_input_path assets/example_sdr.png \
  --output_dir outputs/inference \
  --seed 42
```

Outputs include:

- SDR reconstructions,
- three-channel gain maps,
- HDR `.hdr` frames (BT.2020 gamut, >4 000 nits when `qmax=99`),
- PNG visualisations suitable for paper figures.

Adjust the scheduler or guidance scale to match the paper’s ablation settings when reproducing
metrics such as FHLP/EHL or BRISQUE/NIQE.

## Experiments & Visualisation

The `scripts/stage2/experiments/` and `scripts/inference/experiments/` folders contain the code
used for:

- Gain-map downsampling ablations,
- Scheduler and batch-size sweeps,
- ControlNet-conditioned HDR generation,
- SDR-to-HDRTV conversions on HDRTV1K,
- Rebuttal visualisations.

Each script mirrors the setup from the paper—update the placeholder paths before running.

## Citation

Please cite the ICCV version if you use this repository:

## License

The code will transition to a permissive license (MIT/BSD) upon acceptance. Until then, treat the
repository as “all rights reserved” and contact the authors before redistributing any portion.
