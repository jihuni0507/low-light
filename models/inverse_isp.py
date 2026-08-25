"""Inverse-ISP low-light image synthesis.

The transform models a small camera pipeline:

    sRGB image -> linear sensor values -> exposure/noise -> sRGB image

Inputs are RGB tensors in ``[0, 1]`` with shape ``(3, H, W)`` or
``(B, 3, H, W)``. The module is intentionally dependency-light so it can be
used before image synthesis or independently as a preprocessing step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image


def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    """Convert an sRGB tensor in ``[0, 1]`` to linear sensor intensity."""
    image = image.clamp(0.0, 1.0)
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    """Apply the standard sRGB transfer function to linear intensity."""
    image = image.clamp(0.0, 1.0)
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * image.clamp_min(0.0).pow(1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)


class InverseISP(torch.nn.Module):
    """Synthesize realistic-looking low-light RGB images from rendered images.

    The operation is stochastic during training/inference by default. Set a
    seed in :meth:`forward` for reproducible target-image generation.
    """

    def __init__(
        self,
        exposure_range: tuple[float, float] = (0.08, 0.30),
        shot_noise: float = 0.012,
        read_noise: float = 0.003,
        white_balance_range: tuple[float, float] = (0.96, 1.04),
    ) -> None:
        super().__init__()
        if not 0 < exposure_range[0] <= exposure_range[1] <= 1:
            raise ValueError("exposure_range must satisfy 0 < min <= max <= 1")
        if shot_noise < 0 or read_noise < 0:
            raise ValueError("noise levels must be non-negative")
        if not 0 < white_balance_range[0] <= white_balance_range[1]:
            raise ValueError("white_balance_range must be positive and ordered")

        self.exposure_range = exposure_range
        self.shot_noise = shot_noise
        self.read_noise = read_noise
        self.white_balance_range = white_balance_range

    def forward(
        self,
        image: torch.Tensor,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Return a low-light image with the same shape/device as ``image``."""
        if image.ndim not in (3, 4) or image.shape[-3] != 3:
            raise ValueError("image must have shape (3, H, W) or (B, 3, H, W)")
        if not torch.is_floating_point(image):
            raise TypeError("image must be a floating-point tensor in [0, 1]")
        if not torch.isfinite(image).all():
            raise ValueError("image contains NaN or infinite values")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("image values must be in [0, 1]")

        unbatched = image.ndim == 3
        batch = image.unsqueeze(0) if unbatched else image
        generator = None
        if seed is not None:
            generator = torch.Generator(device=batch.device).manual_seed(seed)

        sensor = srgb_to_linear(batch)
        batch_size = sensor.shape[0]
        random_shape = (batch_size, 1, 1, 1)
        exposure = torch.empty(random_shape, device=batch.device, dtype=batch.dtype)
        exposure = exposure.uniform_(*self.exposure_range, generator=generator)
        gains = torch.empty((batch_size, 3, 1, 1), device=batch.device, dtype=batch.dtype)
        gains = gains.uniform_(*self.white_balance_range, generator=generator)

        sensor = sensor * exposure * gains
        shot_std = self.shot_noise * sensor.clamp_min(0.0).sqrt()
        noise = torch.randn(sensor.shape, device=sensor.device, dtype=sensor.dtype, generator=generator)
        noise = noise * shot_std
        if self.read_noise:
            noise = noise + self.read_noise * torch.randn(
                sensor.shape, device=sensor.device, dtype=sensor.dtype, generator=generator
            )
        sensor = (sensor + noise).clamp(0.0, 1.0)

        output = linear_to_srgb(sensor)
        return output.squeeze(0) if unbatched else output

    def from_pil(self, image: Image.Image, seed: Optional[int] = None) -> Image.Image:
        """Synthesize a PIL image while preserving the RGB image contract."""
        tensor = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)
        tensor = tensor.to(dtype=torch.float32).div(255.0)
        result = self(tensor, seed=seed).mul(255).round().byte()
        return Image.fromarray(result.permute(1, 2, 0).cpu().numpy(), mode="RGB")

    def from_path(self, input_path: Union[str, Path], output_path: Union[str, Path], seed: Optional[int] = None) -> None:
        """Read an image, synthesize its low-light version, and save it."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(input_path) as image:
            self.from_pil(image, seed=seed).save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a low-light image with an inverse ISP transform.")
    parser.add_argument("input", type=Path, help="Input RGB image")
    parser.add_argument("output", type=Path, help="Output low-light image")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible synthesis")
    args = parser.parse_args()
    InverseISP().from_path(args.input, args.output, seed=args.seed)