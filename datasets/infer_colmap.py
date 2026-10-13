import argparse
from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from scipy.spatial.transform import Rotation

from src.config import load_typed_root_config
from src.dataset.shims.crop_shim import apply_crop_shim
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper
from src.misc.image_io import save_image
from src.model.ply_export import export_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a COLMAP target camera from two fixed context views."
    )
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/colmap/render.png"))
    parser.add_argument(
        "--ply-output",
        type=Path,
        default=Path("outputs/colmap/gaussians.ply"),
    )
    parser.add_argument("--experiment", default="re10k")
    parser.add_argument("--context-indices", type=int, nargs=2, default=(0, 7))
    parser.add_argument("--target-index", type=int, default=14)
    parser.add_argument("--near", type=float)
    parser.add_argument("--far", type=float)
    return parser.parse_args()


def read_colmap_text_model(
    model_path: Path,
) -> list[tuple[str, np.ndarray, np.ndarray, int, int, int]]:
    cameras = {}
    with (model_path / "cameras.txt").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split()
            camera_id, model, width, height = (
                int(values[0]),
                values[1],
                int(values[2]),
                int(values[3]),
            )
            params = [float(value) for value in values[4:]]
            if model == "SIMPLE_PINHOLE":
                fx, cx, cy = params
                fy = fx
            elif model == "PINHOLE":
                fx, fy, cx, cy = params
            else:
                raise ValueError(
                    f"Unsupported COLMAP camera model {model!r}; "
                    "undistort the images to PINHOLE first."
                )
            cameras[camera_id] = (width, height, fx, fy, cx, cy)

    images = []
    lines = (model_path / "images.txt").read_text(encoding="utf-8").splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].strip()
        if not line or line.startswith("#"):
            line_index += 1
            continue

        values = line.split(maxsplit=9)
        if len(values) < 10:
            raise ValueError(f"Malformed COLMAP image record: {line}")
        image_id = int(values[0])
        quaternion = np.asarray([float(value) for value in values[1:5]])
        translation = np.asarray([float(value) for value in values[5:8]])
        camera_id = int(values[8])
        image_name = values[9]
        if camera_id not in cameras:
            raise ValueError(
                f"COLMAP image {image_name!r} references missing camera {camera_id}."
            )

        qw, qx, qy, qz = quaternion
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c[:3, 3] = translation
        c2w = np.linalg.inv(w2c).astype(np.float32)
        width, height, fx, fy, cx, cy = cameras[camera_id]
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = fx / width
        intrinsics[1, 1] = fy / height
        intrinsics[0, 2] = cx / width
        intrinsics[1, 2] = cy / height
        images.append((image_name, c2w, intrinsics, width, height, image_id))

        # COLMAP images.txt stores one 2D-point record after each image record.
        line_index += 2

    return sorted(images, key=lambda item: item[0])


def make_views(
    records: list[tuple[str, np.ndarray, np.ndarray, int, int, int]],
    selected: list[int],
    images_dir: Path,
    to_tensor: tf.ToTensor,
    near: float,
    far: float,
    load_images: bool = True,
) -> dict[str, torch.Tensor]:
    images = []
    extrinsics = []
    intrinsics = []
    for index in selected:
        name, c2w, k, width, height, _ = records[index]
        if load_images:
            image_path = images_dir / name
            if not image_path.is_file():
                raise FileNotFoundError(f"COLMAP image file not found: {image_path}")
            with Image.open(image_path) as image_file:
                image = to_tensor(image_file.convert("RGB"))
        else:
            image = torch.zeros((3, height, width), dtype=torch.float32)
        images.append(image)
        extrinsics.append(torch.from_numpy(c2w))
        intrinsics.append(torch.from_numpy(k))

    shapes = {tuple(image.shape[-2:]) for image in images}
    if len(shapes) != 1:
        raise ValueError("Selected COLMAP images must have identical dimensions.")

    return {
        "image": torch.stack(images),
        "extrinsics": torch.stack(extrinsics),
        "intrinsics": torch.stack(intrinsics),
        "near": torch.full((len(selected),), near, dtype=torch.float32),
        "far": torch.full((len(selected),), far, dtype=torch.float32),
        "index": torch.tensor(selected, dtype=torch.int64),
    }


def run() -> None:
    args = parse_args()
    if args.context_indices[0] == args.context_indices[1]:
        raise ValueError("Context indices must refer to two different images.")
    if args.target_index in args.context_indices:
        raise ValueError("The target index must not be one of the context indices.")
    if min(*args.context_indices, args.target_index) < 0:
        raise ValueError("COLMAP image indices must be non-negative.")

    args.colmap_model = args.colmap_model.resolve()
    args.images = args.images.resolve()
    args.checkpoint = args.checkpoint.resolve()
    for required in (
        args.colmap_model / "cameras.txt",
        args.colmap_model / "images.txt",
        args.checkpoint,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Required file not found: {required}")

    repo_root = Path(__file__).resolve().parents[2]
    with hydra.initialize_config_dir(
        config_dir=str(repo_root / "config"),
        version_base=None,
    ):
        cfg_dict = hydra.compose(
            config_name="main",
            overrides=[
                f"+experiment={args.experiment}",
                "mode=test",
            ],
        )
    cfg_dict.checkpointing.load = str(args.checkpoint)

    cfg = load_typed_root_config(cfg_dict)
    if cfg.dataset.view_sampler.num_context_views != 2:
        raise ValueError(
            "This inference script uses a two-context-view checkpoint/config."
        )
    near = cfg.dataset.near if args.near is None else args.near
    far = cfg.dataset.far if args.far is None else args.far
    if near <= 0 or far <= near:
        raise ValueError(f"Expected 0 < near < far, got near={near}, far={far}.")

    records = read_colmap_text_model(args.colmap_model)
    max_index = max(*args.context_indices, args.target_index)
    if max_index >= len(records):
        raise IndexError(
            f"Requested image index {max_index}, but the COLMAP model has "
            f"only {len(records)} registered images."
        )

    set_cfg(cfg_dict)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "losses": get_losses(cfg.loss),
        "step_tracker": None,
    }
    model = ModelWrapper.load_from_checkpoint(
        args.checkpoint,
        **model_kwargs,
        strict=True,
        map_location="cpu",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("MVSplat's splatting_cuda decoder requires a CUDA GPU.")
    device = torch.device("cuda")
    model = model.to(device).eval()

    to_tensor = tf.ToTensor()
    context = make_views(
        records,
        list(args.context_indices),
        args.images,
        to_tensor,
        near,
        far,
    )
    target = make_views(
        records,
        [args.target_index],
        args.images,
        to_tensor,
        near,
        far,
        load_images=False,
    )
    image_shape = tuple(cfg.dataset.image_shape)
    example = apply_crop_shim(
        {"context": context, "target": target},
        image_shape,
    )
    batch = {
        stage: {
            key: value.unsqueeze(0).to(device)
            for key, value in views.items()
        }
        for stage, views in example.items()
    }
    batch["scene"] = [args.colmap_model.parent.name]
    batch = model.data_shim(batch)

    with torch.inference_mode():
        gaussians = model.encoder(
            batch["context"],
            global_step=0,
            deterministic=True,
            scene_names=batch["scene"],
        )
        covariance = gaussians.covariances[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvectors = eigenvectors.clone()
        reflection = torch.linalg.det(eigenvectors) < 0
        eigenvectors[reflection, :, -1] *= -1
        gaussian_scales = eigenvalues.clamp_min(1e-12).sqrt()
        gaussian_rotations = torch.from_numpy(
            Rotation.from_matrix(eigenvectors.detach().cpu().numpy()).as_quat()
        ).to(device=device, dtype=gaussians.means.dtype)
        export_ply(
            batch["context"]["extrinsics"][0, 0],
            gaussians.means[0],
            gaussian_scales,
            gaussian_rotations,
            gaussians.harmonics[0],
            gaussians.opacities[0],
            args.ply_output,
        )
        h, w = batch["context"]["image"].shape[-2:]
        rendered = model.decoder(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        ).color[0, 0]

    save_image(rendered, args.output)
    print(f"Rendered target index {args.target_index} to {args.output.resolve()}")
    print(f"Exported Gaussian PLY to {args.ply_output.resolve()}")
    print(
        "Context indices: "
        f"{args.context_indices[0]}, {args.context_indices[1]} "
        "(indices refer to registered COLMAP images sorted by filename)."
    )


if __name__ == "__main__":
    run()
