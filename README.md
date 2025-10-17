# GM Diffusion

End-to-end pipeline for learning guidance-map driven HDR generation from natural images.
The project is structured around the three phases used in our ICCV 2025 submission:

1. **Stage 1 — Dataset Construction with VAE finetuning**: augment MSCOCO imagery with tone-mapping
   operations to build text/SDR/GM triplets and adapt the base VQGAN with LoRA.
2. **Stage 2 — GM denoiser fine-tuning**: extend Stable Diffusion with a GM branch that
   produces HDR guidance for every SDR sample.
3. **Inference — Paired SDR-GM generation**: run the dual-branch pipeline to synthesise SDR
   renders and their associated guidance maps in parallel.

The repository is now packaged so that reusable components live under `gm_diffusion/` while
executable entry points sit in `scripts/`.

## Repository Layout

```
gm_diffusion/           # Core Python package
├── stage1/             # Augmentations, tone-mapping ops, discriminator
└── pipelines/          # Stable Diffusion pipelines (single & dual UNet variants)
scripts/
├── stage1/             # Stage 1 training CLI
├── stage2/             # Stage 2 training CLI (+ experiments)
└── inference/          # Stage 3 inference CLI (+ experiments)
```

Experimental notebooks and quick-and-dirty prototypes now live under the
`scripts/*/experiments/` subfolders. They ship with placeholder paths (`/path/to/...`) that
should be updated locally if you intend to reproduce the research ablations.

## Getting Started

### Environment

The code targets Python 3.10+ and PyTorch with CUDA 11.8 or newer. The quickest way to get
rolling is to create a virtual environment and install the major dependencies used across
all three stages:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers>=0.23.0 transformers accelerate datasets peft timm wandb opencv-python
```

Install any extra packages required by your hardware setup (e.g. `xformers`, `bitsandbytes`).
`accelerate config` is recommended for preparing distributed/Deepspeed launches.

### Data Prerequisites

Stage 1 expects one or more parquet files containing MSCOCO samples with SDR/GM metadata
columns. See `gm_diffusion/stage1/tone_mapping.py` for the transformations used when
producing these triplets. Each parquet file should minimally include:

| column          | description                                        |
|-----------------|----------------------------------------------------|
| `image`         | image bytes or file path for the SDR frame         |
| `text`          | paired prompt/caption                              |
| `gm_path`       | file path or bytes for the guidance-map target     |

Caching can be handled through either Hugging Face `datasets` (`--dataset_cache_dir`) or a
custom dataloader that writes parquet manifests.

## Stage 1 – VQGAN LoRA Finetuning

Run the Stage 1 trainer through `accelerate`:

```bash
accelerate launch scripts/stage1/train_vqgan_lora.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_metadata data/stage1/train_metadata.parquet \
  --dataset_cache_dir data/cache \
  --resolution 512 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --output_dir outputs/stage1_vqgan
```

Key arguments:

- `--pretrained_model_name_or_path`: base VAE/VQGAN checkpoint.
- `--train_metadata`: one or more parquet files describing the SDR/GM pairs.
- `--bright_tmo`: tone-mapping operator to apply (`hard_clip_tmo`, `linear_scale_tmo`,
  `fix_mulog_tmo` from `gm_diffusion.stage1`).
- `--clip_pixel`: enable adaptive exposure augmentation via
  `RandomExposureAdjust`.

The script automatically wraps the VAE with a configurable LoRA adapter and logs to
TensorBoard/W&B if enabled through `accelerate`.

## Stage 2 – Stable Diffusion GM Branch Finetuning

Stage 2 learns a dual-branch Stable Diffusion UNet that predicts guidance maps conditioned
on the SDR latent. Launch with:

```bash
accelerate launch scripts/stage2/train_gm_unet.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_metadata data/stage2/*.parquet \
  --dataset_cache_dir data/cache \
  --validation_prompt_file configs/validation_prompts.txt \
  --validation_image_dir data/validation_sdr \
  --output_dir outputs/stage2_gm_unet \
  --gradient_accumulation_steps 2 \
  --train_batch_size 2
```

Useful switches:

- `--scheduler_config`: optional override for the diffusion scheduler configuration.
- `--validation_prompt_file` / `--validation_prompts`: provide evaluation prompts.
- `--validation_image_dir`: SDR reference frames used when logging validation samples.
- `--train_metadata`: parquet files containing SDR/GM training labels (same schema as Stage 1
  but typically with paired latent tensors).

Checkpoints are written under `--output_dir` and can be uploaded to the Hub via
`--push_to_hub`.

## Stage 3 – Paired SDR/HDR Inference

Use the helper script to generate SDR renders, guidance maps, and reconstructed HDR frames:

```bash
python scripts/inference/generate_hdr.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --unet_ckpt outputs/stage2_gm_unet/checkpoint-123456 \
  --sdr_input_path assets/example_sdr.png \
  --output_dir outputs/inference \
  --seed 42
```

The script:

1. Loads the base Stable Diffusion components (tokenizer, text encoder, VAE).
2. Swaps in the GM-aware UNet checkpoint.
3. Reconstructs HDR frames via `gm_diffusion.stage1.apply_gm_to_sdr`.

Outputs include SDR reconstructions, predicted guidance maps, HDR `.hdr` dumps, and PNG
visualisations. Adjust `--resolution` or scheduler parameters as needed.

## Experiments & Visualisation

Under `scripts/stage2/experiments/` and `scripts/inference/experiments/` you will find
research prototypes used during the ICCV submission (batch-size studies, scheduler
ablation, rebuttal visualisation). These scripts preserve the original experiment flow but
ship with placeholder paths. They are not required for the main training/inference loop.

## Citation

Coming soon.

## License

This repository will adopt a permissive license (MIT/BSD) before the public release. Until a
license file is added, please treat the code as “all rights reserved” and contact the authors
for redistribution inquiries.
