import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from plyfile import PlyData


def load_gaussians_from_ply(ply_path, device):
    """Load 3DGS PLY fields into the tensor representation used by this renderer."""
    vertex = PlyData.read(str(ply_path))["vertex"]
    property_names = set(vertex.data.dtype.names or ())
    required = {
        "x", "y", "z", "opacity",
        *(f"scale_{index}" for index in range(3)),
        *(f"rot_{index}" for index in range(4)),
        *(f"f_dc_{index}" for index in range(3)),
    }
    missing = sorted(required - property_names)
    if missing:
        raise ValueError(f"PLY is missing required Gaussian properties: {missing}")

    xyz = np.stack([vertex[name] for name in ("x", "y", "z")], axis=-1).copy()
    scaling = np.stack(
        [vertex[f"scale_{index}"] for index in range(3)], axis=-1
    ).copy()
    rotation = np.stack(
        [vertex[f"rot_{index}"] for index in range(4)], axis=-1
    ).copy()
    opacity = np.asarray(vertex["opacity"], dtype=np.float32).reshape(-1, 1).copy()
    features_dc = np.stack(
        [vertex[f"f_dc_{index}"] for index in range(3)], axis=-1
    ).copy()

    rest_names = sorted(
        (name for name in property_names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if len(rest_names) % 3:
        raise ValueError("The number of f_rest properties must be divisible by 3")
    if rest_names:
        # PLY stores the SH rest coefficients channel-major (all R, then G, then B).
        rest = np.stack([vertex[name] for name in rest_names], axis=-1).copy()
        rest = rest.reshape(len(vertex), 3, -1).transpose(0, 2, 1)
    else:
        rest = np.empty((len(vertex), 0, 3), dtype=np.float32)

    return {
        "xyz": torch.as_tensor(xyz, dtype=torch.float32, device=device),
        "scaling": torch.as_tensor(scaling, dtype=torch.float32, device=device),
        "rotation": torch.as_tensor(rotation, dtype=torch.float32, device=device),
        "opacity": torch.as_tensor(opacity, dtype=torch.float32, device=device),
        "features_dc": torch.as_tensor(
            features_dc, dtype=torch.float32, device=device
        ),
        "features_rest": torch.as_tensor(rest, dtype=torch.float32, device=device),
    }


def _get_gaussian_tensors(gaussians, device, scales_are_log, opacities_are_logits):
    if not isinstance(gaussians, Mapping):
        raise TypeError("gaussians must be a mapping of Gaussian tensors")

    def first_present(*names):
        for name in names:
            if name in gaussians and gaussians[name] is not None:
                return gaussians[name]
        raise ValueError(f"Gaussian tensors must include one of: {', '.join(names)}")

    xyz = torch.as_tensor(first_present("xyz", "means"), device=device, dtype=torch.float32)
    scaling = torch.as_tensor(
        first_present("scaling", "scale", "scales"),
        device=device,
        dtype=torch.float32,
    )
    rotation = torch.as_tensor(
        first_present("rotation", "rotations"), device=device, dtype=torch.float32
    )
    opacity = torch.as_tensor(
        first_present("opacity", "opacities"), device=device, dtype=torch.float32
    ).reshape(-1, 1)

    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {tuple(xyz.shape)}")
    if scaling.shape != xyz.shape:
        raise ValueError(
            f"scaling must have shape {tuple(xyz.shape)}, got {tuple(scaling.shape)}"
        )
    if rotation.shape != (xyz.shape[0], 4):
        raise ValueError(
            f"rotation must have shape [N, 4], got {tuple(rotation.shape)}"
        )
    if opacity.shape[0] != xyz.shape[0]:
        raise ValueError("opacity must have one value per Gaussian")
    if scales_are_log:
        scaling = scaling.exp()
    if opacities_are_logits:
        opacity = opacity.sigmoid()

    if "colors" in gaussians and gaussians["colors"] is not None:
        colors = torch.as_tensor(
            gaussians["colors"], device=device, dtype=torch.float32
        )
        if colors.shape != xyz.shape:
            raise ValueError(f"colors must have shape {tuple(xyz.shape)}")
        return xyz, scaling, rotation, opacity, None, colors

    if "features" in gaussians and gaussians["features"] is not None:
        features = torch.as_tensor(
            gaussians["features"], device=device, dtype=torch.float32
        )
        if features.ndim != 3 or features.shape[0] != xyz.shape[0] or features.shape[2] != 3:
            raise ValueError("features must have shape [N, SH coefficients, 3]")
    else:
        dc = torch.as_tensor(
            first_present("features_dc"), device=device, dtype=torch.float32
        )
        if dc.ndim == 2 and dc.shape == xyz.shape:
            dc = dc[:, None, :]
        if dc.shape != (xyz.shape[0], 1, 3):
            raise ValueError("features_dc must have shape [N, 3] or [N, 1, 3]")

        rest_value = gaussians.get("features_rest")
        if rest_value is None:
            rest = torch.empty(
                (xyz.shape[0], 0, 3), device=device, dtype=torch.float32
            )
        else:
            rest = torch.as_tensor(rest_value, device=device, dtype=torch.float32)
            if rest.ndim == 2:
                if rest.shape[0] != xyz.shape[0] or rest.shape[1] % 3:
                    raise ValueError(
                        "2D features_rest must have shape [N, 3 * SH_rest_coefficients]"
                    )
                rest = rest.reshape(xyz.shape[0], 3, -1).transpose(1, 2)
            if rest.ndim != 3 or rest.shape[0] != xyz.shape[0] or rest.shape[2] != 3:
                raise ValueError("features_rest must have shape [N, K, 3]")
        features = torch.cat((dc, rest), dim=1)

    degree = math.isqrt(features.shape[1]) - 1
    if (degree + 1) ** 2 != features.shape[1]:
        raise ValueError(
            f"SH coefficient count must be a square number, got {features.shape[1]}"
        )
    return xyz, scaling, rotation, opacity, features.contiguous(), None


def _camera_matrices(camera, device):
    width = int(camera["width"])
    height = int(camera["height"])
    if width <= 0 or height <= 0:
        raise ValueError("Camera width and height must be positive")

    if "c2w" in camera:
        camera_to_world = torch.as_tensor(
            camera["c2w"], dtype=torch.float32, device=device
        )
        if camera_to_world.shape != (4, 4):
            raise ValueError("Camera c2w must have shape [4, 4]")
    else:
        rotation = torch.as_tensor(
            camera["rotation"], dtype=torch.float32, device=device
        )
        position = torch.as_tensor(
            camera["position"], dtype=torch.float32, device=device
        )
        if rotation.shape != (3, 3) or position.shape != (3,):
            raise ValueError("Camera rotation and position must have shapes [3, 3] and [3]")
        camera_to_world = torch.eye(4, dtype=torch.float32, device=device)
        camera_to_world[:3, :3] = rotation
        camera_to_world[:3, 3] = position

    if "intrinsics" in camera:
        intrinsics = torch.as_tensor(
            camera["intrinsics"], dtype=torch.float32, device=device
        )
        if intrinsics.shape != (3, 3):
            raise ValueError("Camera intrinsics must have shape [3, 3]")
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    else:
        fx = torch.as_tensor(camera["fx"], dtype=torch.float32, device=device)
        fy = torch.as_tensor(camera["fy"], dtype=torch.float32, device=device)
        cx = torch.as_tensor(
            camera.get("cx", width * 0.5), dtype=torch.float32, device=device
        )
        cy = torch.as_tensor(
            camera.get("cy", height * 0.5), dtype=torch.float32, device=device
        )
    if not torch.isfinite(fx) or not torch.isfinite(fy) or fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths fx and fy must be finite and positive")

    near = float(camera.get("znear", 0.01))
    far = float(camera.get("zfar", 100.0))
    if near <= 0 or far <= near:
        raise ValueError("Camera clipping planes must satisfy 0 < znear < zfar")

    projection = torch.zeros((4, 4), dtype=torch.float32, device=device)
    projection[0, 0] = 2.0 * fx / width
    projection[1, 1] = 2.0 * fy / height
    projection[0, 2] = 2.0 * cx / width - 1.0
    projection[1, 2] = 2.0 * cy / height - 1.0
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -(far * near) / (far - near)
    projection[3, 2] = 1.0

    world_to_camera = torch.linalg.inv(camera_to_world)
    view_matrix = world_to_camera.transpose(0, 1).contiguous()
    projection_matrix = projection.transpose(0, 1).contiguous()
    full_projection = view_matrix @ projection_matrix
    return width, height, fx, fy, camera_to_world, view_matrix, full_projection


def render_gaussians(
    gaussians,
    camera,
    background=(0.0, 0.0, 0.0),
    device="cuda",
    scales_are_log=True,
    opacities_are_logits=True,
):
    """Rasterize Gaussian tensors using a camera-to-world matrix and intrinsics.

    Tensor mappings use xyz, scaling, rotation, opacity, and either SH features
    or RGB colors. By default, scaling and opacity are interpreted as log-scales
    and logits, matching the project's Gaussian PLY/model tensors.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    if device.type != "cuda":
        raise RuntimeError("diff_gaussian_rasterization requires a CUDA device")

    xyz, scales, rotations, opacity, shs, colors = _get_gaussian_tensors(
        gaussians, device, scales_are_log, opacities_are_logits
    )
    if xyz.shape[0] == 0:
        raise ValueError("Cannot render an empty Gaussian set")
    if not torch.isfinite(xyz).all() or not torch.isfinite(scales).all():
        raise ValueError("Gaussian positions and scales must be finite")
    if (scales <= 0).any():
        raise ValueError("Gaussian scales must be positive after conversion")

    width, height, fx, fy, camera_to_world, view_matrix, full_projection = (
        _camera_matrices(camera, device)
    )
    bg = torch.as_tensor(background, dtype=torch.float32, device=device)
    if bg.shape != (3,):
        raise ValueError("background must contain exactly three RGB values")

    if shs is not None:
        degree = math.isqrt(shs.shape[1]) - 1
        raster_shs = shs
        precomputed_colors = None
    else:
        degree = 0
        raster_shs = None
        precomputed_colors = colors

    settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=width / (2.0 * float(fx)),
        tanfovy=height / (2.0 * float(fy)),
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=full_projection,
        sh_degree=degree,
        campos=camera_to_world[:3, 3],
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    screen_space_points = torch.zeros_like(xyz, requires_grad=True)
    result = rasterizer(
        means3D=xyz,
        means2D=screen_space_points,
        shs=raster_shs,
        colors_precomp=precomputed_colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    image = result[0].clamp(0.0, 1.0)
    return image


def load_camera_entries(camera_json):
    with Path(camera_json).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    entries = payload.get("cameras") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No camera entries found in {camera_json}")
    return entries


def _look_at_camera(position, target, width, height, fx, fy, uid):
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, world_up)) > 0.98:
        world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return {
        "id": uid,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": width * 0.5,
        "cy": height * 0.5,
        "rotation": np.column_stack((right, up, forward)).tolist(),
        "position": position.tolist(),
    }


def orbit_cameras(entries, frames, radius_scale=1.0, height_offset=0.0):
    if frames <= 0 or radius_scale <= 0:
        raise ValueError("frames and radius_scale must be positive")
    positions = np.asarray([entry["position"] for entry in entries], dtype=np.float64)
    target = positions.mean(axis=0)
    horizontal_offsets = positions[:, (0, 2)] - target[(0, 2)]
    radius = float(np.linalg.norm(horizontal_offsets, axis=1).mean()) * radius_scale
    if radius <= 1e-6:
        radius = float(np.linalg.norm(positions - target, axis=1).mean()) * radius_scale
    if radius <= 1e-6:
        raise ValueError("Cannot infer an orbit radius from camera positions")

    reference = entries[0]
    cameras = []
    for index in range(frames):
        angle = 2.0 * math.pi * index / frames
        position = target + np.array(
            [radius * math.cos(angle), height_offset, radius * math.sin(angle)]
        )
        cameras.append(
            _look_at_camera(
                position,
                target,
                int(reference["width"]),
                int(reference["height"]),
                float(reference["fx"]),
                float(reference["fy"]),
                index,
            )
        )
    return cameras


def main():
    parser = argparse.ArgumentParser(
        description="Render Gaussian PLY/tensors using camera JSON metadata."
    )
    parser.add_argument("--ply", required=True, help="Gaussian PLY file to render")
    parser.add_argument("--camera_json", required=True, help="JSON camera entries")
    parser.add_argument("--output_dir", default="output/rendered")
    parser.add_argument("--mode", choices=("cameras", "orbit"), default="cameras")
    parser.add_argument("--camera_ids", nargs="*", type=int)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--video", help="Optional output video path")
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--height_offset", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    device = torch.device(args.device)
    gaussians = load_gaussians_from_ply(args.ply, device)
    entries = load_camera_entries(args.camera_json)
    by_id = {int(entry.get("id", index)): entry for index, entry in enumerate(entries)}

    if args.mode == "orbit":
        cameras = orbit_cameras(
            entries, args.frames, args.radius_scale, args.height_offset
        )
    elif args.camera_ids:
        missing = sorted(set(args.camera_ids) - by_id.keys())
        if missing:
            raise ValueError(f"Camera ids not found in {args.camera_json}: {missing}")
        cameras = [by_id[camera_id] for camera_id in args.camera_ids]
    else:
        cameras = entries

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video) if args.video else None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        with torch.no_grad():
            for index, camera in enumerate(cameras):
                image = render_gaussians(gaussians, camera, device=device)
                rgb = (
                    image.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
                    * 255.0
                ).astype(np.uint8)
                frame_path = output_dir / f"{index:05d}.png"
                if not cv2.imwrite(
                    str(frame_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ):
                    raise OSError(f"Could not write rendered image: {frame_path}")

                if video_path is not None:
                    if writer is None:
                        height, width = rgb.shape[:2]
                        writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(
                                f"Could not open video writer: {video_path}"
                            )
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()

    print(f"Rendered {len(cameras)} images to {output_dir}")
    if video_path is not None:
        print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()
