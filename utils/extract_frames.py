''' extract_frames.py
For Video Dataset
Extracts frames from a video file and saves them as images in a specified output directory.
'''

import os
import cv2
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_dir, num_views=5, center_frame=None, frame_offset=5):
    '''
    Output directory structure
    output_dir/
        scene1/
            view1.jpg
            view2.jpg
            view3.jpg
            view4.jpg
            view5.jpg
        scene2/
            view1.jpg
            view2.jpg
            view3.jpg
        ...
    '''
        
    if num_views < 1:
        raise ValueError("num_views must be at least 1")
    if frame_offset < 0:
        raise ValueError("frame_offset must be non-negative")

    os.makedirs(f'{output_dir}/images', exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        capture.release()
        raise RuntimeError(f"Video contains no readable frames: {video_path}")

    if center_frame is None:
        center_frame = (frame_count - 1) // 2
    if not 0 <= center_frame < frame_count:
        capture.release()
        raise ValueError(
            f"center_frame must be between 0 and {frame_count - 1}, got {center_frame}"
        )

    half_views = num_views // 2
    offsets = [
        round(index * frame_offset / max(half_views, 1))
        for index in range(-half_views, half_views + 1)
    ]
    frame_indices = [center_frame + offset for offset in offsets]
    if min(frame_indices) < 0 or max(frame_indices) >= frame_count:
        capture.release()
        raise ValueError(
            "The requested centered frame window is outside the video: "
            f"frames {min(frame_indices)}..{max(frame_indices)}, "
            f"valid range 0..{frame_count - 1}"
        )
    saved = 0
    for view_index, frame_index in enumerate(frame_indices, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            continue
        output_path = os.path.join(output_dir, 'images', f"view{view_index}.jpg")
        
        if not cv2.imwrite(output_path, frame):
            capture.release()
            raise RuntimeError(f"Could not write frame: {output_path}")
        saved += 1

    capture.release()
    if saved != num_views:
        raise RuntimeError(
            f"Could only extract {saved}/{num_views} frames from {video_path}"
        )

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video dataset.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the directory with input video files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames.")
    parser.add_argument("--num_views", type=int, default=5, help="Number of views to extract (default: 5).")
    parser.add_argument(
        "--center_frame",
        type=int,
        default=None,
        help="Center frame index for local sampling. Defaults to the middle frame of each video.",
    )
    parser.add_argument(
        "--frame_offset",
        type=int,
        default=5,
        help="Distance from the center frame to the outer views (default: 5).",
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    video_paths = [
        (video_name, os.path.join(args.video_path, video_name))
        for video_name in sorted(os.listdir(args.video_path))
        if video_name.lower().endswith((".mp4", ".avi", ".mov"))
    ]
    progress_bar = tqdm(video_paths, desc="Processing videos", unit="video")
    for video_name, video_path in progress_bar:
        progress_bar.set_postfix_str(video_name, refresh=True)
        scene_name = os.path.splitext(video_name)[0]
        extract_frames(
            video_path,
            os.path.join(args.output_dir, scene_name),
            args.num_views,
            args.center_frame,
            args.frame_offset,
        )
        
    

if __name__ == "__main__":
    main()