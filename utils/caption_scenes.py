''' caption_scenes.py
Generates captions for video scenes using a pre-trained model and saves them in a specified output directory.
'''

import os
import argparse
import json
import PIL
from tqdm import tqdm

def caption_scenes(video_dir, output_path):
    '''
    Generates a dictionary of captions for each scenes, not each views
    '''
    
    captions = {}
    
    for view in tqdm(os.listdir(video_dir), desc="Generating captions..."):
        if not view.lower().endswith((".jpg", ".png")):
            print("Skipping non-image file:", view)
            continue
        else:
            scene_name = os.path.splitext(view)[0]
            
            # Right now creates caption for only first view of each scene,
            # but can be modified if needed
            
            if captions[scene_name] is None:
                # Placeholder for actual caption generation logic
                captions[scene_name] = f"Caption for {scene_name}"

    # Save captions to the output directory
    os.makedirs(output_path, exist_ok=True)
    dict_name = os.path.join(output_path, "captions.json")
    with open(dict_name, 'w') as f:
        json.dump(captions, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Generate captions for video scenes")
    parser.add_argument("--video_dir", help="Path to the input video directory")
    parser.add_argument("--output_path", help="Path to the output directory for captions")
    args = parser.parse_args()

    if not args.video_dir or not args.output_path:
        parser.print_help()
        return

    caption_scenes(args.video_dir, args.output_path)

if __name__ == "__main__":
    main()