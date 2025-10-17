"""
GM Diffusion
================

Utilities for building the HDR guidance-map diffusion system that powers the
three-stage ICCV 2025 pipeline:

1. Stage 1 – dataset preparation and VQGAN pretraining.
2. Stage 2 – Stable Diffusion UNet fine-tuning with guidance maps.
3. Inference – paired SDR/HDR( guidance-map) generation.

The package exposes reusable components so that scripts in ``scripts/`` have a
single import path regardless of the execution stage.
"""

from .stage1.augmentations import RandomExposureAdjust
from .stage1.tone_mapping import (
    apply_gm_to_sdr,
    gamut_compress,
    hard_clip_tmo,
    linear_scale_tmo,
    random_tmo_cuda,
    tmo_cuda,
)

__all__ = [
    "RandomExposureAdjust",
    "apply_gm_to_sdr",
    "gamut_compress",
    "hard_clip_tmo",
    "linear_scale_tmo",
    "random_tmo_cuda",
    "tmo_cuda",
]
