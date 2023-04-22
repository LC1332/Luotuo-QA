import torch
torch.set_default_tensor_type(torch.cuda.HalfTensor)

def get_model(model_name: str, peft_path: str = ""):
    print("Loading model..." + model_name + ("" if peft_path == "" else (" lora:"+peft_path)))
    from transformers import AutoModel
    import torch
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map='auto')
    if peft_path is not None and peft_path != "":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, peft_path)
    return model

from transformers import PreTrainedTokenizer
def gen(model, tokenizer: PreTrainedTokenizer, input_text, max_length=1024):
    ids = tokenizer.encode(
        input_text,
        truncation=True,
        add_special_tokens=True,
    )
    input_ids = torch.LongTensor([ids])
    input_ids = input_ids.to(model.device)
    out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=False,
        temperature=0
    )[0]
    # print(out, tokenizer.pad_token_id)
    out_text = tokenizer.decode(list[int](out)[len(ids):])
    answer = out_text.replace("\nEND", "").strip()
    return answer

def main(
    model_name: str = "THUDM/chatglm-6b",
    model_revision: str = None,
    peft_path: str = "./output",
    test_data_path: str = "test/test-data.json",
    test_data_output_path: str = None,
):
    if test_data_output_path is None or test_data_output_path == "":
        import os
        test_data_output_path = os.path.join(peft_path, "test-data-res.json")
    import json
    instructions = json.load(open(test_data_path, encoding="utf-8"))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision = model_revision)

    import os
    import sys
    import inspect
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir) 
    from train.dataset_tokenize_rows import format_example_list

    with torch.no_grad():
        test_data_output = [[]]*len(instructions)
        for idx, item in enumerate(instructions):
            test_data_output[idx] = {"story": item["story"]}
            example_list = format_example_list(item)
            test_data_output[idx]["questions"] = [[]] * len(example_list)
            for example_idx, example in enumerate(example_list):
                test_data_output[idx]["questions"][example_idx] = {
                    "origin_question": example["origin_question"],
                    "origin_target": example["target"],
                }
        model = None
        model = get_model(model_name)
        for idx, item in enumerate(instructions):
            example_list = format_example_list(item)
            for example_idx, example in enumerate(example_list):
                context = example["context"]
                # print(f"### {idx+1}(origin context):\n{context}")
                answer = gen(model, tokenizer, context)
                print(f"### {idx+1}(origin): ###\n{answer}")
                test_data_output[idx]["questions"][example_idx][model_name] = answer
        # close model
        model = None
        model = get_model(model_name, peft_path)
        for idx, item in enumerate(instructions):
            example_list = format_example_list(item)
            for example_idx, example in enumerate(example_list):
                context = example["context"]
                # print(f"### {idx+1}(lora context):\n{context}")
                answer = gen(model, tokenizer, context)
                print(f"### {idx+1}(lora): ###\n{answer}")
                test_data_output[idx]["questions"][example_idx][model_name+peft_path] = answer
    json.dump(test_data_output, open(test_data_output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
