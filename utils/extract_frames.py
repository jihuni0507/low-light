''' extract_frames.py
For Video Dataset
Extracts frames from a video file and saves them as images in a specified output directory.
'''

import os
import PIL
from PIL import Image
import cv2
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_dir, num_views=3):
    '''
    Expected output directory structure
    output_dir/
        scene1_view1.jpg
        scene1_view2.jpg
        scene1_view3.jpg
        scene2_view1.jpg
        scene2_view2.jpg
        scene2_view3.jpg
        ...
    '''
        
    os.makedirs(output_dir, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video dataset.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the directory with input video files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames.")
    parser.add_argument("--num_views", type=int, default=3, help="Number of views to extract (default: 3).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for video_name, video_path in tqdm([(f, os.path.join(args.video_path, f)) for f in os.listdir(args.video_path) if f.lower().endswith((".mp4", ".avi", ".mov"))], desc="Processing videos"):
        print(f"Processing video: {video_name}")
        extract_frames(video_path, os.path.join(args.output_dir, video_name), args.num_views)
        tqdm.update(1)
    

if __name__ == "__main__":
    main()