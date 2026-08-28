from pathlib import Path
import json
import yaml
import argparse

def create_config(darken_root, gs_root):
    samples = []

    for dark_dir in sorted(darken_root.glob("musharna_*"), key=lambda p: int(p.name.split("_")[-1])):
        scene = dark_dir.name
        gs_dir = gs_root / f"{scene}_5views"
        camera_json = gs_dir / "cameras.json"
        ply = gs_dir / "point_cloud/iteration_30000/point_cloud.ply"

        cameras = json.loads(camera_json.read_text())
        views = []

        for camera in cameras:
            view_name = camera["img_name"]
            image_path = dark_dir / f"target_{view_name}.jpg"

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            views.append({
                "input_image": str(image_path)
            })

        samples.append({
            "id": scene,
            "source_gaussian_ply": str(ply),
            "views": views,
            "camera_json": str(camera_json),
            "camera_ids": [camera["id"] for camera in cameras],
            "prompt": "low-light scene",
        })

    config = {
        "dataset": {
            "name": "musharna_low_light",
            "root": ".",
            "num_views": 5,
            "samples": samples,
            "validation": {
                "require_input_image": True,
                "require_target_gs": False,
                "require_prompt": True,
                "require_matching_gaussian_count": True,
                "require_matching_sh_shape": True,
            },
        }
    }
    
    return config

def main():
    parser = argparse.ArgumentParser(description="YAML dumper for dataset.yaml")
    parser.add_argument("--dark_scenes", type=str, default="/home/work/test2/datasets/darken")
    parser.add_argument("--ply_dir", type=str, default="/home/work/test2/StereoGS/output/LLFF")
    parser.add_argument("--yaml_path", type=str, default="/home/work/test2/low-light/dataset.yaml")
    args = parser.parse_args()
    
    darken_root = Path(args.dark_scenes)
    gs_root = Path(args.ply_dir)
    
    config = create_config(darken_root, gs_root)

    yaml_path = Path(args.yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Created {yaml_path} with {len(config['dataset']['samples'])} scenes")
    
if __name__ == "__main__":
    main()