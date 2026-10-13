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
        exposure_range: tuple[float, float] = (0.001, 0.01),
        noise = True,
        shot_noise: float = 0.012,
        read_noise: float = 0.003,
        white_balance_range: tuple[float, float] = (0.96, 1.04),
    ) -> None:
        super().__init__()
        if not 0 < exposure_range[0] <= exposure_range[1] <= 1:
            raise ValueError("exposure_range must satisfy 0 < min <= max <= 1")
        if not noise:
            self.shot_noise = 0.0
            self.read_noise = 0.0
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
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Return a low-light image with the same shape/device as ``image``."""
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be between 0 and 1")
        if image.ndim not in (3, 4) or image.shape[-3] != 3:
            raise ValueError("image must have shape (3, H, W) or (B, 3, H, W)")
        if not torch.is_floating_point(image):
            raise TypeError("image must be a floating-point tensor in [0, 1]")
        if not torch.isfinite(image).all():
            raise ValueError("image contains NaN or infinite values")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("image values must be in [0, 1]")
        if strength == 0.0:
            return image

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
        exposure = 1.0 - strength * (1.0 - exposure)
        gains = torch.empty((batch_size, 3, 1, 1), device=batch.device, dtype=batch.dtype)
        gains = gains.uniform_(*self.white_balance_range, generator=generator)
        gains = 1.0 + strength * (gains - 1.0)

        sensor = sensor * exposure * gains
        shot_std = strength * self.shot_noise * sensor.clamp_min(0.0).sqrt()
        noise = torch.randn(sensor.shape, device=sensor.device, dtype=sensor.dtype, generator=generator)
        noise = noise * shot_std
        if self.read_noise:
            noise = noise + strength * self.read_noise * torch.randn(
                sensor.shape, device=sensor.device, dtype=sensor.dtype, generator=generator
            )
        sensor = (sensor + noise).clamp(0.0, 1.0)

        output = linear_to_srgb(sensor)
        return output.squeeze(0) if unbatched else output

    def from_pil(
        self,
        image: Image.Image,
        seed: Optional[int] = None,
        strength: float = 1.0,
    ) -> Image.Image:
        """Synthesize a PIL image while preserving the RGB image contract."""
        tensor = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)
        tensor = tensor.to(dtype=torch.float32).div(255.0)
        result = self(tensor, seed=seed, strength=strength).mul(255).round().byte()
        return Image.fromarray(result.permute(1, 2, 0).cpu().numpy(), mode="RGB")

    def from_path(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        seed: Optional[int] = None,
        strength: float = 1.0,
    ) -> None:
        """Read an image, synthesize its low-light version, and save it."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(input_path) as image:
            self.from_pil(image, seed=seed, strength=strength).save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a low-light image with an inverse ISP transform.")
    parser.add_argument("input", type=Path, help="Input RGB image or directory of images")
    parser.add_argument("output", type=Path, help="Output image path or directory")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible synthesis")
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Darkening strength from 0 (no change) to 1 (near-black)",
    )
    parser.add_argument("--noise", type=bool, default=True, help="True/False for shot noise & read noise")
    args = parser.parse_args()
    if not 0.0 <= args.strength <= 1.0:
        parser.error("--strength must be between 0 and 1")

    valid_extensions = {".jpg", ".jpeg", ".png"}
    inverse_isp = InverseISP()
    if args.input.is_file():
        if args.input.suffix.lower() not in valid_extensions:
            parser.error(f"Unsupported image type: {args.input}")
        inverse_isp.from_path(args.input, args.output, seed=args.seed, strength=args.strength)
    elif args.input.is_dir():
        image_paths = sorted(
            path for path in args.input.iterdir()
            if path.is_file() and path.suffix.lower() in valid_extensions
        )
        if not image_paths:
            parser.error(f"No supported images found in {args.input}")

        args.output.mkdir(parents=True, exist_ok=True)
        for index, image_path in enumerate(image_paths):
            seed = None if args.seed is None else args.seed + index
            output_path = args.output / image_path.name
            inverse_isp.from_path(image_path, output_path, seed=seed, strength=args.strength)
            print(f"Saved: {output_path}")
    else:
        parser.error(f"Input does not exist or is not a file/directory: {args.input}")