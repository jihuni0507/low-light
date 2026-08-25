import os
import argparse

from models.inverse_isp import InverseISP

def main():
    parser = argparse.ArgumentParser(description="Generate low-light images with InverseISP.")
    parser.add_argument("--source_dir", default="img/source_images")
    parser.add_argument("--target_dir", default="img/target_images")
    parser.add_argument("--single_dir", default=False, action="store_true", help="If True, treat source_dir as a single directory of images.")
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    inverse_isp = InverseISP()
    
    img_dir = os.path.join(args.source_dir, "images")

    if args.single_dir:
        image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    else:
        image_paths = []
        for root, dirs, files in os.walk(args.source_dir):
            for filename in files:
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, filename))

    for image_path in sorted(image_paths):
        image_name = os.path.basename(image_path)
        output_path = os.path.join(args.target_dir, f"target_{image_name}")
        inverse_isp.from_path(image_path, output_path)
        print(f"Processed {image_name} -> {output_path}")
        
if __name__ == "__main__":
    main()