import argparse
import json
from tqdm import tqdm

def get_context_item(story: str, qa: dict):
    for q in qa['questions']:
        if q == "" or q == " ":
            return None, None, None
    if len(qa['questions']) < 2:
        return None, None, None
    origin_question = qa['questions'][0]
    context = f"""给你下面的文本和问题，请先给出一个对应问题的同义转述，再给出问题的答案。
文本为：{story}
原始问题为：{origin_question}
"""
    target = f"""问题转义为：{qa['questions'][1]}
答案为：{qa['answer']}"""
    return context, origin_question, target

def format_example_list(example: dict) -> list[dict]:
    res = []
    for qa in example['QA']:
        context, origin_question, target = get_context_item(example['story'], qa)
        if context is not None:
            res.append({"context": context, "target": target, "origin_question": origin_question})
    return res

def show_sample_info(example: dict):
    print("### sample info ###")
    print(example)
    # show all keys
    print(example.keys())

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

    show_sample_info(examples[0])
    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w', encoding="utf-8") as f:
        for example in tqdm(examples, desc="formatting.."):
            format = format_example_list(example)
            for item in format:
                f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    main()
