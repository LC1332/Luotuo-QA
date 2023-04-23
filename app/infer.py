import torch
torch.set_default_tensor_type(torch.cuda.HalfTensor)

example_story = """北京时间2月13日凌晨,2023年ATP250达拉斯站男单决赛。中国球员吴易昺先输一盘后挽救4个赛点并兑现第5个冠军点,最终以6(4)-7/7-6(3)/7-6(12)逆转惊险击败赛会5号种子、美国大炮伊斯内尔,就此改写历史,成为公开赛年代首位夺得ATP巡回赛男单冠军的中国大陆球员,并创造中国大陆球员的男单最高排名!

第一盘比赛,吴易昺在第12局错过了一个盘点,并最终抢七惜败;第二盘则挽救一个赛点后抢七局3-0领先开局,且以7-6(3)扳回一盘;第三盘决胜盘,在关键的第9局15-40落后情况下凭借连续的高质量发球逆转保发,之后比赛再次进入抢七,抢七局依然胶着,吴易昺又挽救了3个赛点,并兑现了自己的第5个冠军点,就此锁定冠军!历史性一刻到来时,吴易昺瞬间躺倒在地。全场比赛,伊斯内尔轰出了44记Ace球,但最终在主场依然输给了吴易昺。

凭借具有突破意义的这一冠,吴易昺在本周入账250个积分和112125美元的冠军奖金,在周一最新一期的男单排名榜单上,创中国大陆男网历史新高排名—第58位。根据比赛计划,吴易昺原本要出战本周进行的ATP250德拉海滩站,不过在达拉斯夺冠后,吴易昺因身体疲劳退出本站赛事,他的签位由幸运落败者约翰森替代。"""

example_question = "这场赛事中，谁是伊斯内尔的有力竞争者？"

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

def format_context(story, question):
    return f"""给你下面的文本和问题，请先给出一个对应问题的同义转述，再给出问题的答案。
文本为：{story}
原始问题为：{question}
"""

def infer_gen(model, tokenizer, context):
    out = gen(model, tokenizer, context)
    question_as = out.split("答案为:")[0].split("问题转义为:")[1]
    answer = out.split("答案为:")[1]
    return out, question_as, answer

def infer(model, tokenizer, story, question, origin_model = None):
    context = format_context(story, question)
    origin_out = ""
    if origin_model is not None:
        origin_out = gen(origin_model, tokenizer, context)
    out, question_as, answer = infer_gen(model, tokenizer, context)
    
    context_v2 = format_context(story, question_as)
    out_v2, question_as_v2, answer_v2 = infer_gen(model, tokenizer, context_v2)
    
    print(f"### {context}: ###\n Origin: {origin_out}\n Lora: {out}\n Lora^2: {out_v2}\n")
    return origin_out, question_as, answer, question_as_v2, answer_v2

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
