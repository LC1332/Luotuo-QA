import argparse
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train-data.json")
    parser.add_argument("--save_path", type=str, default="data/train-data.jsonl")
    args = parser.parse_args()

    with open(args.data_path, encoding="utf-8") as f:
        examples = json.load(f)
    if isinstance(examples, dict):
        if "data" in examples:
            examples = examples["data"]

    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w', encoding="utf-8") as f:
        for example in tqdm(examples, desc="saving jsonl..."):
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
