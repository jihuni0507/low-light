from datasets import load_dataset

url = (
    "https://huggingface.co/datasets/"
    "DavidYan2001/RealEstate10K/resolve/main/dataset/test.tar.gz"
)
dataset = load_dataset(
    "webdataset",
    data_files={"test": url},
    split="test",
    streaming=True,
)

subset = dataset.take(5)

for i, sample in enumerate(subset):
    print(f"--- Sample {i} ---")
    print(sample.keys())