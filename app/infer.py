import torch
torch.set_default_tensor_type(torch.cuda.HalfTensor)
from transformers import PreTrainedTokenizer
from peft import PeftModel

example_story = """北京时间2月13日凌晨,2023年ATP250达拉斯站男单决赛。中国球员吴易昺先输一盘后挽救4个赛点并兑现第5个冠军点,最终以6(4)-7/7-6(3)/7-6(12)逆转惊险击败赛会5号种子、美国大炮伊斯内尔,就此改写历史,成为公开赛年代首位夺得ATP巡回赛男单冠军的中国大陆球员,并创造中国大陆球员的男单最高排名!

第一盘比赛,吴易昺在第12局错过了一个盘点,并最终抢七惜败;第二盘则挽救一个赛点后抢七局3-0领先开局,且以7-6(3)扳回一盘;第三盘决胜盘,在关键的第9局15-40落后情况下凭借连续的高质量发球逆转保发,之后比赛再次进入抢七,抢七局依然胶着,吴易昺又挽救了3个赛点,并兑现了自己的第5个冠军点,就此锁定冠军!历史性一刻到来时,吴易昺瞬间躺倒在地。全场比赛,伊斯内尔轰出了44记Ace球,但最终在主场依然输给了吴易昺。

凭借具有突破意义的这一冠,吴易昺在本周入账250个积分和112125美元的冠军奖金,在周一最新一期的男单排名榜单上,创中国大陆男网历史新高排名—第58位。根据比赛计划,吴易昺原本要出战本周进行的ATP250德拉海滩站,不过在达拉斯夺冠后,吴易昺因身体疲劳退出本站赛事,他的签位由幸运落败者约翰森替代。"""

example_question = "谁会参加ATP250德拉海滩站？"

def get_model(model_name: str, peft_path: str = ""):
    print("Loading model..." + model_name + ("" if peft_path == "" else (" lora:"+peft_path)))
    from transformers import AutoModel
    import torch
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map='auto')
    if peft_path is not None and peft_path != "":
        model = PeftModel.from_pretrained(model, peft_path)
    return model

def format_context(story, question):
    return f"""给你下面的文本和问题，请给出问题的答案。
文本为：{story}
问题为：{question}
"""

def infer_gen(model, tokenizer, context):
    out = gen(model, tokenizer, context)
    answer = out.split("答案为:")[1]
    return out, answer

def infer(model, tokenizer, story, question, origin_model = None):
    context = format_context(story, question)
    origin_out = ""
    if origin_model is not None:
        origin_out = gen(origin_model, tokenizer, context)
    out, answer = infer_gen(model, tokenizer, context)
    
    print(f"### {context}: ###\n Origin: {origin_out}\n Lora: {out}\n")
    return origin_out, answer

def infer_yield(model, tokenizer, story, question, origin_model = None):
    context = format_context(story, question)
    origin_out = ""
    if origin_model is not None:
        origin_out = gen(origin_model, tokenizer, context)
    yield origin_out, ""
    out, answer = infer_gen(model, tokenizer, context)
    print(f"### {context}: ###\n Origin: {origin_out}\n Lora: {out}\n")
    yield origin_out, answer

def question_answer_infer(model, tokenizer: PreTrainedTokenizer, story, question, max_length=2048):
    append_text = f"""答案为:"""

    input_token_ids = tokenizer.encode(format_context(story, question))
    input_ids = torch.LongTensor([input_token_ids]).to(model.device)
    append_token_ids = tokenizer.encode(append_text)
    append_ids = torch.LongTensor([append_token_ids]).to(model.device)
    out = continue_generate(model, input_ids, append_ids, 
        max_length = max_length, 
        do_sample=True, 
        top_p=0.4, 
        temperature=0.95, 
        logits_processor=None,
    )[0]
    out_text = tokenizer.decode(list[int](out)[len(input_token_ids) + len(append_token_ids):])
    answer = out_text.replace("\nEND", "").strip()
    print(f"question_answer_infer: ###{answer}###")

    return answer

def gen(model: PeftModel, tokenizer: PreTrainedTokenizer, input_text, max_length=2048):
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

import copy
import warnings

import torch
import torch.utils.checkpoint
from torch import nn
from typing import Optional, List, Callable

from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig

from transformers.utils import logging
logger = logging.get_logger(__name__)

@torch.no_grad()
def continue_generate(
        model,
        input_ids: torch.Tensor,
        append_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        **kwargs,
):
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            logger.warn(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = model._get_logits_warper(generation_config)
    
    input_ids = torch.cat([input_ids, append_ids], dim=-1)
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break
    return input_ids

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
