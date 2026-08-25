import os
import argparse
import torch
import torchvision.transforms.functional as TF
from PIL import Image

def enhance_dark_data(
    image: torch.Tensor,
    exposure_boost: float = 4.0,
    chroma_restore_ratio: float = 0.5,
    gamma: float = 2.2
) -> torch.Tensor:

    # 1. Inverse Chroma Suppression
    luma = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    luma = luma.unsqueeze(0).repeat(3, 1, 1)
    if chroma_restore_ratio < 1.0:
        restored_srgb = (image - chroma_restore_ratio * luma) / (1.0 - chroma_restore_ratio)
        restored_srgb = torch.clamp(restored_srgb, 0.0, 1.0)
    else:
        restored_srgb = image

    # 2. sRGB to Linear Space
    dark_linear = torch.pow(restored_srgb, gamma)
    
    # 3. Exposure Boost
    enhanced_linear = dark_linear * exposure_boost

    # 4. Linear to sRGB: Tone Mapping Restoration + Gamma Correction
    # 더 고급의 ISP 모사를 원한다면 Reinhard Tone Mapping 등을 고려할 수 있음
    enhanced_linear_clamped = torch.clamp(enhanced_linear, 0.0, 1.0)
    enhanced_srgb = torch.pow(enhanced_linear_clamped, 1.0/gamma)

    return enhanced_srgb

def main():
    parser = argparse.ArgumentParser(description="Enhance Los-Light images to Normal-Light (Inverse SplatBright Method)")
    parser.add_argument("--input_dir", type=str, required=True, help='Directory containing dark images')
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save enhanced images")
    parser.add_argument("--boost", type=float, default=4.0, help="Exposure boost factor (default: 4.0)")
    parser.add_argument("--chroma_restore", type=float, default=5.0, help="Chroma suppresion ratio used in darkening (default: 0.5)")
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma value (default: 2.2)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    valid_exts = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG')
    files = [f for f in os.listdir(args.input_dir) if f.endswith(valid_exts)]

    if not files:
        print(f"No valid images found in {args.input_dir}!")
        return

    for file_name in files:
        img_path = os.path.join(args.input_dir, file_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = TF.to_tensor(img)

            enhanced_tensor = enhance_dark_data(img_tensor, args.boost,
            args.chroma_restore, args.gamma)

            enhanced_img = TF.to_pil_image(enhanced_tensor)
            out_path = os.path.join(args.output_dir, file_name)

            enhanced_img.save(out_path)

            print(f"    [+] Saved: {out_path}")
        
        except Exception as e:
            print(f"    [-] Failed to process {file_name}: {e}")

if __name__ == '__main__':
    main()