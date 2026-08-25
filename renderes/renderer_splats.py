import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

STEREOGS_ROOT = Path(__file__).resolve().parents[2] / "StereoGS"
if str(STEREOGS_ROOT) not in sys.path:
    sys.path.insert(0, str(STEREOGS_ROOT))

from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel


def camera_from_entry(entry, device):
    width = int(entry["width"])
    height = int(entry["height"])
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = np.asarray(entry["rotation"], dtype=np.float64)
    camera_to_world[:3, 3] = np.asarray(entry["position"], dtype=np.float64)
    world_to_camera = np.linalg.inv(camera_to_world)
    fovx = 2.0 * math.atan(width / (2.0 * float(entry["fx"])))
    fovy = 2.0 * math.atan(height / (2.0 * float(entry["fy"])))
    return Camera(
        colmap_id=int(entry.get("id", 0)),
        R=world_to_camera[:3, :3].T,
        T=world_to_camera[:3, 3],
        FoVx=fovx,
        FoVy=fovy,
        image=torch.zeros((3, height, width), device=device),
        image_name=entry.get("img_name", str(entry.get("id", 0))),
        uid=int(entry.get("id", 0)),
        data_device=str(device),
    )


def load_camera_entries(camera_json):
    with open(camera_json, "r", encoding="utf-8") as file:
        entries = json.load(file)
    if not entries:
        raise ValueError(f"No cameras found in {camera_json}")
    return entries


def look_at_camera(position, target, width, height, fx, fy, device, uid):
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
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = np.stack([right, up, forward])
    world_to_camera[:3, 3] = -world_to_camera[:3, :3] @ position
    return Camera(
        colmap_id=uid,
        R=world_to_camera[:3, :3].T,
        T=world_to_camera[:3, 3],
        FoVx=2.0 * math.atan(width / (2.0 * fx)),
        FoVy=2.0 * math.atan(height / (2.0 * fy)),
        image=torch.zeros((3, height, width), device=device),
        image_name=f"orbit_{uid:05d}",
        uid=uid,
        data_device=str(device),
    )


def orbit_cameras(entries, frames, device, radius_scale=1.0, height_offset=0.0):
    positions = np.asarray([entry["position"] for entry in entries], dtype=np.float64)
    target = positions.mean(axis=0)
    horizontal = positions[:, :2] - target[:2]
    radius = np.linalg.norm(horizontal, axis=1).mean() * radius_scale
    if radius <= 1e-6:
        radius = np.linalg.norm(positions - target, axis=1).mean() * radius_scale
    reference = entries[0]
    width = int(reference["width"])
    height = int(reference["height"])
    fx = float(reference["fx"])
    fy = float(reference["fy"])
    cameras = []
    for index in range(frames):
        angle = 2.0 * math.pi * index / frames
        position = target + np.array(
            [radius * math.cos(angle), radius * math.sin(angle), height_offset]
        )
        cameras.append(look_at_camera(position, target, width, height, fx, fy, device, index))
    return cameras


def render_image(camera, gaussians, pipeline, background):
    rendering = render(camera, gaussians, pipeline, background, train=False, dropout_factor=0.0)
    image = rendering["render"].detach().clamp(0.0, 1.0)
    return (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Render an injected StereoGS PLY from JSON cameras or an orbit path.")
    parser.add_argument("--ply", required=True, help="PLY to render, usually injected_gaussians.ply")
    parser.add_argument("--camera_json", required=True, help="StereoGS cameras.json")
    parser.add_argument("--output_dir", default="output/rendered", help="Directory for PNG frames")
    parser.add_argument("--mode", choices=["cameras", "orbit"], default="cameras")
    parser.add_argument("--video", default="", help="Optional MP4 output path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=120, help="Orbit video frame count")
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--height_offset", type=float, default=0.0)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the StereoGS rasterizer.")
    device = torch.device(args.device)
    entries = load_camera_entries(args.camera_json)
    if args.mode == "cameras":
        cameras = [camera_from_entry(entry, device) for entry in entries]
    else:
        cameras = orbit_cameras(entries, args.frames, device, args.radius_scale, args.height_offset)

    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(args.ply)
    gaussians = gaussians.to(device) if hasattr(gaussians, "to") else gaussians
    pipeline = type("Pipeline", (), {"convert_SHs_python": False, "compute_cov3D_python": False, "debug": False})()
    background = torch.zeros(3, device=device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.video:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    writer = None

    with torch.no_grad():
        for index, camera in enumerate(cameras):
            image = render_image(camera, gaussians, pipeline, background)
            frame_path = output_dir / f"{index:05d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if args.video:
                if writer is None:
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(
                        args.video,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer: {args.video}")
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if writer is not None:
        writer.release()
    print(f"Rendered {len(cameras)} frames to {output_dir}")
    if args.video:
        print(f"Saved video to {args.video}")


if __name__ == "__main__":
    main()
