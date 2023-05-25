import argparse
import json
from tqdm import tqdm

import datasets
import transformers
import copy
from transformers import PreTrainedTokenizer


IGNORE_INDEX = -100

def get_context_item(story: str, qa: dict):
    for q in qa['questions']:
        if q == "" or q == " ":
            return None, None, None
    if len(qa['questions']) < 2:
        return None, None, None
    origin_question = qa['questions'][0]
    context = f"""给你下面的文本和问题，请给出问题的答案。
文本为：{story}
问题为：{origin_question}
"""
    target = f"""答案为：{qa['answer']}"""
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

def preprocess(tokenizer: PreTrainedTokenizer, config, example, max_seq_length):
    context_tokens = tokenizer.encode(
        example["context"],
        max_length=max_seq_length,
        truncation=True,
    )
    target_tokens = tokenizer.encode(
        example["target"],
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False,
    )
    return dict(
        input_ids=context_tokens + target_tokens + [config.eos_token_id],
        seq_len=len(context_tokens),
    )

def process_jsonl_lines(examples, tokenizer: PreTrainedTokenizer, config, max_seq_length, skip_overlength=False):
    for example in tqdm(examples):
        feature = preprocess(tokenizer, config, example, max_seq_length)
        if skip_overlength and len(feature["input_ids"]) > max_seq_length:
            continue
        feature["input_ids"] = feature["input_ids"][:max_seq_length]
        yield feature

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="data/train-data.jsonl")
    parser.add_argument("--save_path", type=str, default="data/train")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="THUDM/chatglm-6b")
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading model and tokenizer from {args.model_name}\n")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        args.model_name, trust_remote_code=True, device_map='auto')

    jsonl_lines = read_jsonl(args.jsonl_path)

    if len(jsonl_lines) > 0:
        show_sample_info(json.loads(jsonl_lines[0]))

    examples = []
    for line in tqdm(jsonl_lines, desc="decoding jsonl examples..."):
        examples += format_example_list(json.loads(line))

    dataset = datasets.Dataset.from_generator(
        lambda examples: process_jsonl_lines(examples, tokenizer, config, args.max_seq_length, args.skip_overlength),
        num_proc=args.num_proc,
        gen_kwargs={"examples": examples},
    )
    dataset.save_to_disk(args.save_path)

if __name__ == "__main__":
    main()
