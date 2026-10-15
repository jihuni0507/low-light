from datasets import load_dataset

dataset = load_dataset("DavidYan2001/RealEstate10K", split="test", streaming=True)

# Fetch the first 3 scenes
few_scenes = list(dataset.take(3))

# Access the data
for idx, scene in enumerate(few_scenes):
    print(f"Scene {idx}:")
    print(f"Keys: {scene.keys()}")
    # 'image' contains the PIL Image, and 'label' or path has the metadata