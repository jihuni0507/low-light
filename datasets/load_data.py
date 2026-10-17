from datasets import load_dataset

dataset = load_dataset("Wouter01/re10ksmalltest", split="train")

subset = dataset.take(5)

for i, sample in enumerate(subset):
    print(f"--- Sampe {i} ---")
    print(sample.keys())
    