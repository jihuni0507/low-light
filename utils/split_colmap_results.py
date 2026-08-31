import argparse
import os
import shutil

def split_colmap_text(colmap_dir, output_dir, image_dir, scene_name='scene', keep_every_n=10):
    os.makedirs(output_dir, exist_ok=True)
    
    ### 1) Parsing images.txt ###
    with open(os.path.join(colmap_dir, "images.txt"), "r") as f:
        img_lines = f.readlines()
        
    img_data = {}
    header_lines_img = []
    
    idx = 0
    while idx < len(img_lines):
        line = img_lines[idx].strip()
        if line.startswith("#"):
            header_lines_img.append(line + '\n')
            idx += 1
            continue
        if not line:
            idx += 1
            continue
        
        line1 = line.split()
        line2 = img_lines[idx+1].strip().split()
        img_data[int(line1[0])] = (line1, line2)
        idx += 2
        
    sorted_ids = sorted(img_data.keys())
    print(f"Total images: {len(sorted_ids)} -> Splitting into {keep_every_n} groups...")
    
    for offset in range(keep_every_n):
        kept_img_ids = set(sorted_ids[offset::keep_every_n])
        
        # 3DGS 표준 폴더 구조 생성
        base_dataset_dir = os.path.join(output_dir, f"{scene_name}_{offset}")
        current_sparse_dir = os.path.join(base_dataset_dir, "sparse", "0")
        current_image_dir = os.path.join(base_dataset_dir, "images")
        
        os.makedirs(current_sparse_dir, exist_ok=True)
        os.makedirs(current_image_dir, exist_ok=True)
    
        ### 2) Parsing Points3D.txt ###
        with open(os.path.join(colmap_dir, "points3D.txt"), "r") as f:
            pts_lines = f.readlines()
            
        valid_pts = set()
        new_pts_lines = []
        
        for line in pts_lines:
            if line.startswith("#"):
                new_pts_lines.append(line)
                continue
            parts = line.strip().split()
            if not parts:
                continue
            
            pt_id = int(parts[0])
            tracks = parts[8:] # TRACKS: [IMAGE_ID, POINT2D_IDX, IMAGE_ID, POINT2D_IDX, ...]
            
            new_tracks = []
            for i in range(0, len(tracks), 2):
                img_id = int(tracks[i])
                if img_id in kept_img_ids:
                    new_tracks.extend([tracks[i], tracks[i+1]])
                    
            if len(new_tracks) > 0:
                valid_pts.add(pt_id)
                new_line = " ".join(parts[:8] + new_tracks) + "\n"
                new_pts_lines.append(new_line)
                    
        # 텍스트 파일은 sparse/0 내부에 저장
        with open(os.path.join(current_sparse_dir, "points3D.txt"), "w") as f:
            f.writelines(new_pts_lines)
            
        ### 3) Updated Images, Points and Copy Image Files ###
        new_img_lines = []
        used_cam_ids = set()
        
        for img_id in kept_img_ids:
            l1, original_l2 = img_data[img_id] 
            l2 = original_l2.copy() 
            
            used_cam_ids.add(int(l1[8])) # camera_id 추출
            
            # --- 이미지 파일 복사 로직 시작 ---
            img_name = " ".join(l1[9:]) # 이미지 파일명 추출 (공백 포함된 이름 고려)
            src_img_path = os.path.join(image_dir, img_name)
            dst_img_path = os.path.join(current_image_dir, img_name)
            
            if os.path.exists(src_img_path):
                # 하위 폴더가 있을 경우를 대비해 디렉토리 생성
                os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                shutil.copy2(src_img_path, dst_img_path)
            else:
                print(f"Warning: Image file not found - {src_img_path}")
            # --- 이미지 파일 복사 로직 끝 ---

            new_img_lines.append(" ".join(l1) + "\n")
            
            for i in range(2, len(l2), 3): # l2 : X Y POINT3D_ID X Y POINT3D_ID ...
                pt3d = int(l2[i])
                if pt3d != -1 and pt3d not in valid_pts:
                    l2[i] = "-1" # referring blank space
                    
            new_img_lines.append(" ".join(l2) + "\n")
            
        with open(os.path.join(current_sparse_dir, "images.txt"), "w") as f:
            f.writelines(header_lines_img)
            f.writelines(new_img_lines)
            
        ### 4) Parsing Cameras.txt ###
        with open(os.path.join(colmap_dir, "cameras.txt"), "r") as f:
            cam_lines = f.readlines()
            
        new_cam_lines = []
        for line in cam_lines:
            if line.startswith("#"):
                new_cam_lines.append(line)
                continue
            
            parts = line.strip().split()
            if not parts:
                continue
            
            if int(parts[0]) in used_cam_ids:
                new_cam_lines.append(line)
        
        with open(os.path.join(current_sparse_dir, "cameras.txt"), "w") as f:
            f.writelines(new_cam_lines)  

def main():
    parser = argparse.ArgumentParser(description="Parses COLMAP results into 3DGS sparse datasets (with images)")
    parser.add_argument("--colmap_dir", type=str, required=True, help="Directory with original COLMAP text files")
    parser.add_argument("--image_dir", type=str, help="Directory with original images")
    parser.add_argument("--output_dir", type=str, default='/home/work/test2/datasets')
    parser.add_argument("--scene", type=str, default='scene', help="Prefix name of the output scenes")
    parser.add_argument("--num_split", type=int, default=5, help="Number of splits (e.g. 5 means 1/5 sparse views)")
    args = parser.parse_args()
    
    if args.image_dir:
        original_img_dir = args.image_dir
    else:
        original_img_dir = f'/home/work/test2/datasets/{args.scene}/images'
        
    if not os.path.exists(original_img_dir):
        raise ValueError(f"Original Images Not found from {original_img_dir}")
    
    split_colmap_text(args.colmap_dir, args.output_dir, original_img_dir, scene_name=args.scene,
                      keep_every_n=args.num_split)
    
if __name__ == "__main__":
    main()