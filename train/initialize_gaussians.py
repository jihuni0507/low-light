import torch
import numpy as np
import torch.nn.functional as F

from plyfile import PlyData

C0 = 0.28209479177387814

def ply_to_tensor(ply_path):
    """Read baseline Gaussian properties from a 3DGS PLY into CPU tensors."""
    vertex = PlyData.read(ply_path)["vertex"]
    property_names = set(vertex.data.dtype.names or ())
    required_names = {
        "x", "y", "z", "opacity",
        *(f"scale_{i}" for i in range(3)),
        *(f"rot_{i}" for i in range(4)),
        *(f"f_dc_{i}" for i in range(3)),
    }
    missing = sorted(required_names - property_names)
    if missing:
        raise ValueError(f"PLY is missing required Gaussian properties: {missing}")

    xyz = np.stack([vertex[name] for name in ("x", "y", "z")], axis=-1).copy()
    opacity = np.asarray(vertex["opacity"], dtype=np.float32).reshape(-1, 1).copy()
    scale = np.stack([vertex[f"scale_{i}"] for i in range(3)], axis=-1).copy()
    rotation = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=-1).copy()
    features_dc = np.stack(
        [vertex[f"f_dc_{i}"] for i in range(3)], axis=-1
    ).copy()

    extra_sh_names = sorted(
        (name for name in property_names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    features_extra = None
    if extra_sh_names:
        features_extra = np.stack(
            [vertex[name] for name in extra_sh_names], axis=-1
        ).copy()
        if features_extra.shape[1] % 3 != 0:
            raise ValueError("The number of f_rest properties must be divisible by 3")

    return {
        "xyz": torch.from_numpy(xyz).float(),
        "opacity": torch.from_numpy(opacity).float(),
        "scale": torch.from_numpy(scale).float(),
        "rotation": torch.from_numpy(rotation).float(),
        "features_dc": torch.from_numpy(features_dc).float(),
        "features_extra": (
            torch.from_numpy(features_extra).float()
            if features_extra is not None else None
        ),
    }


def _project_visible_gaussians(xyz, camera, image):
    """Project Gaussian centers and approximate visibility with a center z-buffer."""
    height, width = image.shape[-2:]
    if int(camera["height"]) != height or int(camera["width"]) != width:
        raise ValueError(
            f"Image dimensions {(height, width)} do not match camera dimensions "
            f"{(camera['height'], camera['width'])}"
        )

    camera_to_world_rotation = torch.as_tensor(
        camera["rotation"], dtype=xyz.dtype, device=xyz.device
    )
    camera_position = torch.as_tensor(
        camera["position"], dtype=xyz.dtype, device=xyz.device
    )
    if camera_to_world_rotation.shape != (3, 3) or camera_position.shape != (3,):
        raise ValueError("Camera metadata must contain 3x3 rotation and 3D position")

    camera_xyz = (xyz - camera_position) @ camera_to_world_rotation
    depth = camera_xyz[:, 2]
    fx, fy = float(camera["fx"]), float(camera["fy"])
    cx = float(camera.get("cx", width * 0.5))
    cy = float(camera.get("cy", height * 0.5))
    if fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths fx and fy must be positive")

    u = fx * camera_xyz[:, 0] / depth.clamp_min(1e-8) + cx
    v = fy * camera_xyz[:, 1] / depth.clamp_min(1e-8) + cy
    in_frame = (
        (depth > 1e-8)
        & (u >= 0)
        & (u <= width - 1)
        & (v >= 0)
        & (v <= height - 1)
    )

    visible = torch.zeros_like(in_frame)
    candidates = torch.nonzero(in_frame, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return torch.zeros((xyz.shape[0], 3), device=xyz.device), visible

    pixel_x = u[candidates].round().long()
    pixel_y = v[candidates].round().long()
    pixel_id = pixel_y * width + pixel_x
    nearest_depth = torch.full(
        (height * width,), float("inf"), dtype=depth.dtype, device=depth.device
    )
    nearest_depth.scatter_reduce_(
        0, pixel_id, depth[candidates], reduce="amin", include_self=True
    )
    visible[candidates] = depth[candidates] <= nearest_depth[pixel_id] + 1e-6

    grid_x = 2.0 * u / max(width - 1, 1) - 1.0
    grid_y = 2.0 * v / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, 1, -1, 2)
    sampled = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, :, 0, :].transpose(0, 1)
    sampled = sampled * visible[:, None]
    return sampled, visible


def initialize_gaussians(sample, device):
    """Build per-Gaussian RGB inputs from the two visible input views."""
    gaussians = {
        name: tensor.to(device)
        if isinstance(tensor, torch.Tensor) else tensor
        for name, tensor in ply_to_tensor(sample["ply_path"]).items()
    }
    xyz = gaussians["xyz"]
    base_rgb = (gaussians["features_dc"] * C0 + 0.5).clamp(0.0, 1.0)
    opacity = torch.sigmoid(gaussians["opacity"][:, 0])

    color_sum = torch.zeros_like(base_rgb)
    weight_sum = torch.zeros((xyz.shape[0],), dtype=xyz.dtype, device=device)
    visible_views = torch.zeros_like(weight_sum)
    for view in sample["input_views"]:
        image = view["image"].to(device=device, dtype=xyz.dtype)
        sampled_rgb, visible = _project_visible_gaussians(
            xyz, view["camera"], image
        )
        weights = opacity * visible.to(opacity.dtype)
        color_sum += sampled_rgb * weights[:, None]
        weight_sum += weights
        visible_views += visible.to(visible_views.dtype)

    observed = weight_sum > 1e-8
    sampled_rgb = color_sum / weight_sum.clamp_min(1e-8)[:, None]
    initial_rgb = torch.where(observed[:, None], sampled_rgb, base_rgb)
    gaussians.update({
        "base_rgb": base_rgb,
        "initial_rgb": initial_rgb.transpose(0, 1).unsqueeze(0),
        "observed_mask": observed,
        "color_confidence": (visible_views / len(sample["input_views"])).clamp(0, 1),
    })
    return gaussians