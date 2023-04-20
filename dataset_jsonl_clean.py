import argparse
import json
from tqdm import tqdm
import re

"""
an example of the jsonl file:
{
    "story": "xxxx",
    "idx": 0,
    "QA": [
        {
            "answer": "一个仆人。",
            "questions": [
                "谁收到了他们的母亲的信息？",
                "是谁被一个传递消息的仆人告知回家？",
                "谁被告知有一封由马车送来的信？",
                "谁被他们母亲的信息要求立即离开？",
                "谁被一位仆人打断了，告知他们母亲的消息？"
            ]
        },
        {
            "answer": "一条给我留言的信息。",
            "questions": [
                "这个仆人的信息是针对谁发出的？",
                "2) 由仆人所传递的信息要求谁回家？",
                "3) 是哪个人的计划因信件的到来而受影响？",
                "4）主人公收到的信息中提到了谁的离开建议？",
                "5）谁通过仆人的消息知道了信件的到来？"
            ]
        }
    ]
}
"""

blacklistKeyword = [
    "抱歉，",
    "作为AI",
    "as an AI",
    "Sorry,",
    "AI language model",
    "language model AI",
    "language AI model",
    "不能完成这个请求",
    "为您提供",
    # "帮我翻译",
    # "需要翻译",
    # "翻译结果",
    # "翻译是",
    "翻译",
    "translation",
    "translated"
]

def check_by_blacklist_keyword(content: str) -> bool:
    content = content.strip().lower()
    for keyword in blacklistKeyword:
        keyword = keyword.strip().lower()
        if keyword in content:
            return False
    return True

def clean_example(example: dict) -> list[dict]:
    res = {}
    resQA = []
    for qa in example['QA']:
        questions = []
        for q in qa['questions']:
            q = q.strip()
            # reg remove 1. or 1） or 1、 or 2. or 2） or 2、
            q = re.sub('^\d+[\.()）]\s*', '', q)
            q = q.strip()
            if q == "" or "？" not in q or not check_by_blacklist_keyword(q):
                continue
            questions.append(q)
        if len(questions) < 2:
            continue
        if not check_by_blacklist_keyword(qa['answer']):
            continue
        resQA.append({
            "answer": qa['answer'].strip(),
            "questions": questions,
        })

    if len(example['story']) > 0 and check_by_blacklist_keyword(example['story']) and len(resQA) > 0:
        res['story'] = example['story']
        res['QA'] = resQA
    else:
        return []
    return [res]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train-data.json")
    parser.add_argument("--save_path", type=str, default="data/train-data.jsonl")
    args = parser.parse_args()
    
    from dataset_tokenize_rows import read_jsonl, show_sample_info

    jsonl_lines = read_jsonl(args.data_path)

    if len(jsonl_lines) > 0:
        show_sample_info(json.loads(jsonl_lines[0]))

    examples = []
    for line in tqdm(jsonl_lines, desc="cleaning jsonl examples..."):
        examples += clean_example(json.loads(line))

    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w', encoding="utf-8") as f:
        for example in tqdm(examples, desc="saving cleaned examples jsonl..."):
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
