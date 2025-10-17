"""
Data augmentation utilities used during stage 1 dataset preparation.
"""

from __future__ import annotations

import random
from typing import Dict, Tuple, Union

import torch


class RandomExposureAdjust:
    """
    Simulate camera exposure variations by applying an inverse camera curve,
    discretisation, and gamma correction. Works on batched or single images.
    """

    def __init__(self, gamma: float = 2.2, prob: float = 1.0):
        self.gamma = gamma
        self.prob = prob
        self.exposure_levels = torch.tensor([0.1, 0.25, 0.5, 1.0, 4.0, 8.0, 16.0], dtype=torch.float32)

    def hdr_to_ldr(self, img: torch.Tensor, exposure: float) -> torch.Tensor:
        img = torch.clamp(img * exposure, 0.0, 1.0)
        return torch.pow(img, 1.0 / self.gamma)

    @staticmethod
    def sample_camera_curve() -> Tuple[float, float]:
        n = float(torch.clamp(torch.normal(mean=0.65, std=0.1, size=()), 0.4, 0.9))
        sigma = float(torch.clamp(torch.normal(mean=0.6, std=0.1, size=()), 0.4, 0.8))
        return n, sigma

    @staticmethod
    def apply_inv_sigmoid_curve(y: torch.Tensor, n: float, sigma: float) -> torch.Tensor:
        return torch.pow((sigma * y) / (1 + sigma - y + 1e-8), 1.0 / n)

    @staticmethod
    def discretize_to_uint16(img: torch.Tensor) -> torch.Tensor:
        max_int = 2**16 - 1
        return torch.clamp(img * max_int, 0, max_int).round() / max_int

    def __call__(
        self,
        imgs: torch.Tensor,
        *,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        if random.random() > self.prob:
            return (imgs, {"exposure": 1.0, "n": 1.0, "sigma": 0.0}) if return_metadata else imgs

        exposure = float(self.exposure_levels[torch.randint(len(self.exposure_levels), (1,))])
        n, sigma = self.sample_camera_curve()

        is_batched = imgs.dim() == 4
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.dim() != 4:
            raise ValueError("RandomExposureAdjust expects a tensor with shape (C,H,W) or (N,C,H,W)")
        if imgs.dtype != torch.float32:
            raise TypeError(f"RandomExposureAdjust expects float32 tensors, received {imgs.dtype}")

        linear_img = self.apply_inv_sigmoid_curve(imgs, n, sigma)
        linear_img = self.discretize_to_uint16(linear_img)
        ldr_img = self.hdr_to_ldr(linear_img, exposure)

        if not is_batched:
            ldr_img = ldr_img.squeeze(0)

        if return_metadata:
            metadata = {"exposure": exposure, "n": n, "sigma": sigma}
            return ldr_img, metadata
        return ldr_img

    def __repr__(self) -> str:  # pragma: no cover - simple debug helper
        return (
            f"{self.__class__.__name__}(gamma={self.gamma}, prob={self.prob}, "
            f"exposure_levels={self.exposure_levels.tolist()})"
        )


def _demo() -> None:  # pragma: no cover - debug helper
    """
    CLI demo invoked with: ``python -m gm_diffusion.stage1.augmentations``.
    Generates a random tensor and prints metadata for sanity checks.
    """
    sample = torch.rand(3, 256, 256)
    augment = RandomExposureAdjust()
    adjusted, meta = augment(sample, return_metadata=True)
    print("Adjustment metadata:", meta)
    print("Input stats:", sample.min().item(), sample.max().item())
    print("Output stats:", adjusted.min().item(), adjusted.max().item())


if __name__ == "__main__":  # pragma: no cover
    _demo()
