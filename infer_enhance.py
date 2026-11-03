import argparse
from pathlib import Path

import torch
from PIL import Image
from plyfile import PlyData

from models.enhance_gaussian import gaussian_enhance_net
from render.renderer_splats import (
    load_camera_entries,
    load_gaussians_from_ply,
    render_gaussians,
)

C0 = 0.28209479177387814


def enhance_gaussians(gaussians, network, chunk_size):
    """Enhance the DC SH color of each Gaussian while preserving other fields."""
    features_dc = gaussians["features_dc"]
    if features_dc.ndim != 2 or features_dc.shape[1] != 3:
        raise ValueError(
            f"features_dc must have shape [N, 3], got {tuple(features_dc.shape)}"
        )
    if features_dc.shape[0] == 0:
        raise ValueError("Cannot enhance an empty Gaussian set")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    rgb = (features_dc * C0 + 0.5).clamp(0.0, 1.0)
    enhanced_chunks = []
    for start in range(0, rgb.shape[0], chunk_size):
        rgb_chunk = rgb[start:start + chunk_size]
        _, enhanced_rgb, _ = network(rgb_chunk.transpose(0, 1).unsqueeze(0))
        enhanced_chunks.append(enhanced_rgb[0].transpose(0, 1))

    enhanced_gaussians = dict(gaussians)
    enhanced_rgb = torch.cat(enhanced_chunks, dim=0)
    enhanced_gaussians["features_dc"] = (enhanced_rgb - 0.5) / C0
    return enhanced_gaussians


def load_enhance_network(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Enhancement checkpoint must contain a state dictionary")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model_state_dict must be a state dictionary")

    network = gaussian_enhance_net().to(device)
    network.load_state_dict(state_dict)
    network.eval()
    return network


def save_enhanced_ply(source_path, output_path, enhanced_gaussians):
    ply_data = PlyData.read(str(source_path))
    vertices = ply_data["vertex"].data
    enhanced_dc = enhanced_gaussians["features_dc"].detach().cpu().numpy()
    if len(vertices) != enhanced_dc.shape[0]:
        raise ValueError(
            "Enhanced Gaussian count does not match the source PLY vertex count"
        )

    property_names = set(vertices.dtype.names or ())
    for channel in range(3):
        property_name = f"f_dc_{channel}"
        if property_name not in property_names:
            raise ValueError(f"Source PLY is missing required property {property_name}")
        vertices[property_name] = enhanced_dc[:, channel]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ply_data.write(str(output_path))


def match_camera_image_size(camera, images_dir):
    image_name = camera.get("image_name") or camera.get("img_name")
    if not image_name:
        raise ValueError(
            "Camera metadata must include image_name or img_name to match "
            "the source image size"
        )

    image_path = images_dir / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Source camera image not found: {image_path}")

    with Image.open(image_path) as source_image:
        source_width, source_height = source_image.size

    camera_width = int(camera["width"])
    camera_height = int(camera["height"])
    if camera_width <= 0 or camera_height <= 0:
        raise ValueError(
            f"Camera dimensions must be positive, got "
            f"{camera_width}x{camera_height}"
        )
    if source_width <= 0 or source_height <= 0:
        raise ValueError(
            f"Source image dimensions must be positive, got "
            f"{source_width}x{source_height}: {image_path}"
        )

    if (source_width, source_height) == (camera_width, camera_height):
        return camera

    adjusted_camera = dict(camera)
    scale_x = source_width / camera_width
    scale_y = source_height / camera_height
    adjusted_camera["width"] = source_width
    adjusted_camera["height"] = source_height
    for key in ("fx", "cx"):
        if key in adjusted_camera:
            adjusted_camera[key] *= scale_x
    for key in ("fy", "cy"):
        if key in adjusted_camera:
            adjusted_camera[key] *= scale_y
    if "intrinsics" in adjusted_camera:
        intrinsics = [list(row) for row in adjusted_camera["intrinsics"]]
        intrinsics[0][0] *= scale_x
        intrinsics[0][2] *= scale_x
        intrinsics[1][1] *= scale_y
        intrinsics[1][2] *= scale_y
        adjusted_camera["intrinsics"] = intrinsics
    return adjusted_camera


def render_views(gaussians, cameras, output_dir, device, images_dir=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths = []
    for camera in cameras:
        if images_dir is not None:
            camera = match_camera_image_size(camera, images_dir)
        camera_id = int(camera.get("id", len(rendered_paths)))
        image = render_gaussians(gaussians, camera, device=device)
        image_array = (
            image.permute(1, 2, 0)
            .mul(255)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        output_path = output_dir / f"render_{camera_id:06d}.png"
        Image.fromarray(image_array, mode="RGB").save(output_path)
        rendered_paths.append(output_path)
    return rendered_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Enhance a baseline Gaussian PLY and render it with camera JSON "
            "or a COLMAP text model."
        )
    )
    parser.add_argument(
        "--baseline_gaussian",
        required=True,
        help="Path to the baseline 3D Gaussian PLY file",
    )
    parser.add_argument(
        "--checkpoint",
        default="snapshots/latest.pth",
        help="Trained enhance_network checkpoint",
    )
    camera_input = parser.add_mutually_exclusive_group(required=True)
    camera_input.add_argument(
        "--camera_json",
        help="Camera entries as a JSON list or an object containing 'cameras'",
    )
    camera_input.add_argument(
        "--colmap_model",
        help="COLMAP text model directory containing cameras.txt and images.txt",
    )
    parser.add_argument(
        "--images_dir",
        help=(
            "Directory containing the original camera images; defaults to the "
            "images directory next to --colmap_model"
        ),
    )
    parser.add_argument(
        "--camera_ids",
        nargs="*",
        type=int,
        help="Optional camera ids to render; by default all cameras are rendered",
    )
    parser.add_argument("--output_dir", default="output/enhance_inference")
    parser.add_argument(
        "--output_ply",
        help="Enhanced PLY output path (default: <output_dir>/enhanced_gaussians.ply)",
    )
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    baseline_path = Path(args.baseline_gaussian).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    camera_path = Path(args.camera_json or args.colmap_model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_ply_path = (
        Path(args.output_ply).resolve()
        if args.output_ply
        else output_dir / "enhanced_gaussians.ply"
    )

    for input_path, label in (
        (baseline_path, "Baseline Gaussian PLY"),
        (checkpoint_path, "Enhancement checkpoint"),
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"{label} not found: {input_path}")
    if args.camera_json and not camera_path.is_file():
        raise FileNotFoundError(f"Camera JSON not found: {camera_path}")
    if args.colmap_model and not camera_path.is_dir():
        raise NotADirectoryError(f"COLMAP model directory not found: {camera_path}")
    images_dir = (
        Path(args.images_dir).resolve()
        if args.images_dir
        else camera_path.parent / "images" if args.colmap_model
        else None
    )
    if images_dir is not None and not images_dir.is_dir():
        raise NotADirectoryError(f"Source images directory not found: {images_dir}")
    if baseline_path == output_ply_path:
        raise ValueError("The enhanced PLY output path must differ from the input PLY")
    if args.chunk_size <= 0:
        parser.error("--chunk_size must be positive")

    cameras = (
        load_camera_entries(camera_path)
        if args.camera_json
        else None
    )
    if args.colmap_model:
        camera_sidecar = baseline_path.with_suffix(".cameras.json")
        if not camera_sidecar.is_file():
            raise FileNotFoundError(
                "Camera poses in a COLMAP model are in the original scene "
                "coordinate frame, while the MVSplat-exported Gaussian PLY uses "
                "a centered, scaled, and rotated frame. Re-export the baseline "
                f"PLY to create its camera sidecar ({camera_sidecar}), or pass "
                "a camera JSON already transformed into the PLY coordinate frame "
                "with --camera_json."
            )
        cameras = load_camera_entries(camera_sidecar)
    if cameras is None:
        raise RuntimeError("No camera entries were loaded")
    if args.camera_ids is not None:
        cameras_by_id = {
            int(camera.get("id", index)): camera
            for index, camera in enumerate(cameras)
        }
        missing_ids = sorted(set(args.camera_ids) - cameras_by_id.keys())
        if missing_ids:
            raise ValueError(
                f"Camera ids not found in {camera_path}: {missing_ids}"
            )
        cameras = [cameras_by_id[camera_id] for camera_id in args.camera_ids]
        if not cameras:
            parser.error("--camera_ids must contain at least one camera id")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Gaussian rendering requires CUDA and diff_gaussian_rasterization; "
            "select an available CUDA device with --device."
        )

    gaussians = load_gaussians_from_ply(baseline_path, device)
    network = load_enhance_network(checkpoint_path, device)
    with torch.inference_mode():
        enhanced_gaussians = enhance_gaussians(
            gaussians, network, args.chunk_size
        )
        save_enhanced_ply(baseline_path, output_ply_path, enhanced_gaussians)
        rendered_paths = render_views(
            enhanced_gaussians, cameras, output_dir, device, images_dir
        )

    print(f"Enhanced Gaussian PLY: {output_ply_path}")
    for rendered_path in rendered_paths:
        print(f"Rendered image: {rendered_path}")


if __name__ == "__main__":
    main()
