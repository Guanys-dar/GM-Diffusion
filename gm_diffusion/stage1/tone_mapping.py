"""
Tone-mapping operators and utility transforms used throughout stage 1 and 2.
"""

from __future__ import annotations

import math
import random
from typing import Tuple

import torch


def linear_scale_tmo(img: torch.Tensor, qmax: float) -> torch.Tensor:
    """
    Scale an HDR tensor back to [0, 1] by dividing with the peak luminance.
    """
    return img / (qmax + 1)


def hard_clip_tmo(hdr_img: torch.Tensor, qmax: float) -> torch.Tensor:
    """
    Clamp HDR values to [0, 1]; disregards qmax but keeps signature for API compatibility.
    """
    del qmax
    return torch.clamp(hdr_img, 0, 1)


def fix_mulog_tmo(hdr_img: torch.Tensor, qmax: float) -> torch.Tensor:
    """
    Logarithmic tone mapping with a fixed mu parameter.
    """
    hdr_img = hdr_img / (qmax + 1)
    mu = 500
    tm = torch.log1p(mu * hdr_img) / math.log1p(mu)
    return torch.clamp(tm, 0, 1)


def tmo_cuda(hdr_img: torch.Tensor) -> torch.Tensor:
    """
    CUDA-friendly logarithmic tone mapping used for data augmentation.
    """
    hdr_img = torch.clamp(hdr_img / 10, 0, 1)
    if not torch.all((0 <= hdr_img) & (hdr_img <= 1)):
        raise ValueError("HDR image values should be in the range [0, 1]")
    mu = 5_000.0
    return torch.log1p(mu * hdr_img) / math.log1p(mu)


def random_tmo_cuda(hdr_img: torch.Tensor, qmax: float) -> torch.Tensor:
    """
    Sample a tone-mapping curve with a random logarithmic scale.
    """
    hdr_img = hdr_img / (qmax + 1)
    mu = random.uniform(500, 5_000)
    tm = torch.log1p(mu * hdr_img) / math.log1p(mu)
    return torch.clamp(tm, 0, 1)


def apply_gm_to_sdr(
    gm: torch.Tensor,
    sdr: torch.Tensor,
    qmax: float = 9,
    eps: float = 1 / 64,
) -> torch.Tensor:
    """
    Lift an SDR tensor to HDR using a guidance-map (GM) prediction.
    """
    sdr_linear = torch.clamp(sdr, 0, 1) ** 2.2
    hdr = (sdr_linear + eps) * (1 + gm * qmax) - eps
    return torch.clamp(hdr, 0, qmax + 1)


def gamut_compress(tmo_hdr_img: torch.Tensor) -> torch.Tensor:
    """
    BT2020 → BT709 gamut compression for batched HDR tensors (B, C, H, W).
    """
    conversion = torch.tensor(
        [
            [1.660491, -0.587641, -0.072850],
            [-0.124550, 1.132900, -0.008349],
            [-0.018151, -0.100579, 1.118730],
        ],
        device=tmo_hdr_img.device,
        dtype=tmo_hdr_img.dtype,
    ).t()
    img = tmo_hdr_img.permute(0, 2, 3, 1)
    img = torch.matmul(img, conversion)
    img = img.permute(0, 3, 1, 2)
    return torch.clamp(img, 0, 1)


__all__ = [
    "linear_scale_tmo",
    "hard_clip_tmo",
    "fix_mulog_tmo",
    "tmo_cuda",
    "random_tmo_cuda",
    "apply_gm_to_sdr",
    "gamut_compress",
]
