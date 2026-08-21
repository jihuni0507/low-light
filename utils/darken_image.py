import os
import argparse

from models.inverse_isp import InverseISP

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", default="test2/low-light/img/source_images")
parser.add_argument("--target_dir", default="test2/low-light/img/target_images")
args = parser.parse_args()

def main():
    # Ensure the target directory exists
    os.makedirs(args.target_dir, exist_ok=True)

    source_images = os.listdir(args.source_dir)

    for image_name, image_path in [(name, os.path.join(args.source_dir, name)) for name in source_images]:
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load the image
        img = InverseISP.load_image(image_path)

        # Apply the inverse ISP to generate a low-light image
        inverse_isp = InverseISP()
        low_light_img = inverse_isp(img.unsqueeze(0)).squeeze(0)

        # Save the low-light image
        output_path = os.path.join(args.target_dir, f"target_{image_name}")
        InverseISP.save_image(low_light_img, output_path)
        print(f"Processed {image_name} -> {output_path}")
        
if __name__ == "__main__":
    main()