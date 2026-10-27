import argparse
import shutil
from pathlib import Path

if __package__:
    from .inverse_isp import InverseISP
else:
    from inverse_isp import InverseISP


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def main(
    input_dir: Path,
    output_dir: Path,
    exposure_range: tuple[float, float] = (0.001, 0.01),
    noise: bool = True,
    shot_noise: float = 0.012,
    read_noise: float = 0.003,
    white_balance_range: tuple[float, float] = (0.96, 1.04),
    seed: int | None = None,
    strength: float = 1.0,
) -> None:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if output_dir != input_dir:
        try:
            output_dir.relative_to(input_dir)
        except ValueError:
            pass
        else:
            raise ValueError(
                "Output directory must be outside the input directory, "
                "or be the same directory to create sibling *_darked scenes."
            )

    scenes = sorted(
        scene
        for scene in input_dir.iterdir()
        if scene.is_dir()
        and not scene.name.endswith("_darked")
        and (scene / "images").is_dir()
    )
    if not scenes:
        raise ValueError(f"No scene directories containing an images folder found in {input_dir}")

    scene_images: dict[Path, list[Path]] = {}
    for scene in scenes:
        if not (scene / "sparse").is_dir():
            raise FileNotFoundError(f"COLMAP sparse model not found: {scene / 'sparse'}")
        output_scene = output_dir / f"{scene.name}_darked"
        if output_scene.exists():
            raise FileExistsError(
                f"Output scene already exists: {output_scene}. "
                "Remove or rename the existing output scene before running again."
            )
        image_paths = sorted(
            image_path
            for image_path in (scene / "images").rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No supported images found in {scene / 'images'}")
        scene_images[scene] = image_paths

    inverse_isp = InverseISP(
        exposure_range=exposure_range,
        noise=noise,
        shot_noise=shot_noise,
        read_noise=read_noise,
        white_balance_range=white_balance_range,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_index = 0
    for scene in scenes:
        image_paths = scene_images[scene]
        output_scene = output_dir / f"{scene.name}_darked"
        shutil.copytree(scene / "sparse", output_scene / "sparse")
        for image_path in image_paths:
            relative_path = image_path.relative_to(scene / "images")
            output_image_path = output_scene / "images" / relative_path
            image_seed = None if seed is None else seed + image_index
            inverse_isp.from_path(
                image_path,
                output_image_path,
                seed=image_seed,
                strength=strength,
            )
            image_index += 1
            print(f"Saved: {output_image_path}")

        print(f"Copied COLMAP camera model unchanged: {output_scene / 'sparse'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply inverse ISP to images in COLMAP scenes and copy their sparse models unchanged."
    )
    parser.add_argument(
        "--input_dir",
        "--image_dir",
        dest="input_dir",
        type=Path,
        required=True,
        help="Parent directory containing scene folders",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for the processed scenes",
    )
    parser.add_argument(
        "--exposure_range",
        type=float,
        nargs=2,
        default=(0.001, 0.01),
        metavar=("MIN", "MAX"),
        help="Exposure range for low-light synthesis",
    )
    parser.add_argument("--noise", dest="noise", action="store_true", default=True, help="Enable noise (default)")
    parser.add_argument("--no-noise", dest="noise", action="store_false", help="Disable shot and read noise")
    parser.add_argument("--shot_noise", type=float, default=0.012, help="Shot noise level")
    parser.add_argument("--read_noise", type=float, default=0.003, help="Read noise level")
    parser.add_argument(
        "--white_balance_range",
        type=float,
        nargs=2,
        default=(0.96, 1.04),
        metavar=("MIN", "MAX"),
        help="White balance gain range",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible synthesis")
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Darkening strength from 0 (unchanged) to 1 (full effect)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.strength <= 1.0:
        parser.error("--strength must be between 0 and 1")

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        exposure_range=(args.exposure_range[0], args.exposure_range[1]),
        noise=args.noise,
        shot_noise=args.shot_noise,
        read_noise=args.read_noise,
        white_balance_range=(args.white_balance_range[0], args.white_balance_range[1]),
        seed=args.seed,
        strength=args.strength,
    )
