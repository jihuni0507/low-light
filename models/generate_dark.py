import os
import argparse
import torch
import torchvision.transforms.functional as TF
from PIL import Image

def generate_splatbright_dark_data(
    image: torch.Tensor, 
    sky_mask: torch.Tensor, 
    darkness_factor: float = 4.0, 
    chroma_suppress_ratio: float = 0.5,
    gamma: float = 2.2
) -> torch.Tensor:
    # 1. sRGB to Linear Space
    linear_img = torch.pow(image, gamma)
    
    # 2. Exposure Drop
    dark_linear = linear_img / darkness_factor
    
    # 3. ISP-Tone Compression
    dark_srgb = torch.pow(torch.clamp(dark_linear, 0.0, 1.0), 1.0 / gamma)
    
    # 4. Chroma Suppression
    luma = 0.299 * dark_srgb[0] + 0.587 * dark_srgb[1] + 0.114 * dark_srgb[2]
    luma = luma.unsqueeze(0).repeat(3, 1, 1)
    dark_desaturated = (1.0 - chroma_suppress_ratio) * dark_srgb + chroma_suppress_ratio * luma
    
    # 5. Soft Sky Masking
    final_dark_img = (1.0 - sky_mask) * dark_desaturated + sky_mask * image
    
    return torch.clamp(final_dark_img, 0.0, 1.0)

def main():
    parser = argparse.ArgumentParser(description="Generate Low-Light dataset from Normal-Light images (SplatBright method)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing original images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save darkened images")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory containing sky masks (optional)")
    parser.add_argument("--darkness", type=float, default=4.0, help="Exposure drop factor (default: 4.0)")
    parser.add_argument("--chroma", type=float, default=0.5, help="Chroma suppression ratio 0.0~1.0 (default: 0.5)")
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma value (default: 2.2)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG')
    files = [f for f in os.listdir(args.input_dir) if f.endswith(valid_exts)]
    
    if not files:
        print(f"No valid images found in {args.input_dir}")
        return
        
    print(f"Found {len(files)} images. Starting conversion...")
    
    for file_name in files:
        img_path = os.path.join(args.input_dir, file_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = TF.to_tensor(img)
            
            # 하늘 마스크 처리 (없으면 0으로 초기화하여 전체 영역 어둡게 적용)
            mask_tensor = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2]))
            if args.mask_dir:
                # 마스크 파일 확장자 매칭 시도
                mask_base = os.path.splitext(file_name)[0]
                mask_path = None
                for ext in valid_exts:
                    temp_path = os.path.join(args.mask_dir, mask_base + ext)
                    if os.path.exists(temp_path):
                        mask_path = temp_path
                        break
                        
                if mask_path:
                    mask = Image.open(mask_path).convert('L')
                    mask_tensor = TF.to_tensor(mask)
                else:
                    print(f"  [Warning] Mask for {file_name} not found, defaulting to zero mask.")
            
            # 어두운 이미지 생성
            dark_tensor = generate_splatbright_dark_data(
                img_tensor, mask_tensor, args.darkness, args.chroma, args.gamma
            )
            
            # 저장
            dark_img = TF.to_pil_image(dark_tensor)
            out_path = os.path.join(args.output_dir, file_name)
            dark_img.save(out_path)
            print(f"  [+] Saved: {out_path}")
            
        except Exception as e:
            print(f"  [-] Failed to process {file_name}: {e}")

if __name__ == '__main__':
    main()