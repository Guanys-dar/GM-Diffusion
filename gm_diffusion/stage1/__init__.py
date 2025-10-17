"""
Stage 1 utilities: dataset augmentation, tone-mapping operations, and
lightweight discriminators used during VQGAN training.
"""

from .augmentations import RandomExposureAdjust
from .discriminator import Discriminator
from .tone_mapping import (
    apply_gm_to_sdr,
    fix_mulog_tmo,
    gamut_compress,
    hard_clip_tmo,
    linear_scale_tmo,
    random_tmo_cuda,
    tmo_cuda,
)

__all__ = [
    "RandomExposureAdjust",
    "Discriminator",
    "apply_gm_to_sdr",
    "fix_mulog_tmo",
    "gamut_compress",
    "hard_clip_tmo",
    "linear_scale_tmo",
    "random_tmo_cuda",
    "tmo_cuda",
]
