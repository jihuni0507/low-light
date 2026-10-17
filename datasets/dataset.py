import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset


class GaussianSceneDataset(Dataset):
    def __init__(self, dataset_yaml: str):
        self.dataset_yaml = Path(dataset_yaml).resolve()

        with self.dataset_yaml.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        dataset_config = config["dataset"]
        self.root = Path(dataset_config.get("root", "."))
        if not self.root.is_absolute():
            self.root = (self.dataset_yaml.parent / self.root).resolve()

        self.samples = dataset_config["samples"]
        if not self.samples:
            raise ValueError("dataset.samples must contain at least one scene")

        for i, sample in enumerate(self.samples):
            if not sample.get("id"):
                raise ValueError(f"Scene at index {i} is missing 'id'")
            if not sample.get("source_gaussian_ply"):
                raise ValueError(f"Scene {sample['id']!r} is missing 'source_gaussian_ply'")
            if not sample.get("camera_json"):
                raise ValueError(f"Scene {sample['id']!r} is missing 'camera_json'")

            input_views = sample.get("input_views")
            if not isinstance(input_views, list) or len(input_views) != 2:
                raise ValueError(
                    f"Scene {sample['id']!r} must provide exactly two input_views"
                )
            if not isinstance(sample.get("render_view"), dict):
                raise ValueError(f"Scene {sample['id']!r} is missing render_view")

            for key in ("source_gaussian_ply", "camera_json"):
                path = self._resolve_path(sample[key])
                if not path.is_file():
                    raise FileNotFoundError(f"Scene {sample['id']!r}: {path}")

            for view in input_views:
                image_value = view.get("image", view.get("input_image"))
                if not image_value or "camera_id" not in view:
                    raise ValueError(
                        f"Each input view in scene {sample['id']!r} needs image "
                        "and camera_id"
                    )
                image_path = self._resolve_path(image_value)
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Scene {sample['id']!r}: image not found: {image_path}"
                    )
            if "camera_id" not in sample["render_view"]:
                raise ValueError(
                    f"Scene {sample['id']!r} render_view needs camera_id"
                )

    def _resolve_path(self, path: str) -> Path:
        path = Path(path)
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        with self._resolve_path(sample["camera_json"]).open(
            "r", encoding="utf-8"
        ) as file:
            camera_entries = json.load(file)

        camera_by_id = {int(camera["id"]): camera for camera in camera_entries}
        input_views = []

        for view_config in sample["input_views"]:
            camera_id = int(view_config["camera_id"])
            if camera_id not in camera_by_id:
                raise ValueError(
                    f"Camera id {camera_id} not found for scene {sample['id']!r}"
                )

            image_path = self._resolve_path(
                view_config.get("image", view_config.get("input_image"))
            )
            camera = camera_by_id[camera_id]
            image = self._load_image(image_path)

            if image.shape[-2:] != (int(camera["height"]), int(camera["width"])):
                raise ValueError(
                    f"Image size {tuple(image.shape[-2:])} does not match camera "
                    f"size {(camera['height'], camera['width'])}: {image_path}"
                )

            input_views.append({
                "image": image,
                "image_path": str(image_path),
                "camera_id": camera_id,
                "camera": camera,
            })

        render_camera_id = int(sample["render_view"]["camera_id"])
        if render_camera_id not in camera_by_id:
            raise ValueError(
                f"Render camera id {render_camera_id} not found for scene "
                f"{sample['id']!r}"
            )

        return {
            "scene_id": str(sample["id"]),
            "ply_path": str(self._resolve_path(sample["source_gaussian_ply"])),
            "camera_json": str(self._resolve_path(sample["camera_json"])),
            "input_views": input_views,
            "render_view": {
                "camera_id": render_camera_id,
                "camera": camera_by_id[render_camera_id],
            },
        }


def scene_collate_fn(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Gaussian counts and view counts can differ between scenes.
    return batch