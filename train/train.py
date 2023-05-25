from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os

from transformers import PreTrainedTokenizer
def preprocess(tokenizer: PreTrainedTokenizer, config, context, target, max_seq_length):
    context_tokens = tokenizer.encode(
        context,
        max_length=max_seq_length,
        truncation=True,
    )
    target_tokens = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False,
    )
    return dict(
        input_ids=context_tokens + target_tokens + [config.eos_token_id],
        seq_len=len(context_tokens),
    )


def get_context_item(story: str, q1, answer):
    if q1 == "" or answer == "":
        return None, None, None
    origin_question = q1
    context = f"""给你下面的文本和问题，请给出问题的答案。
文本为：{story}
问题为：{origin_question}
"""
    target = f"""答案为：{answer}"""
    return context, origin_question, target


@dataclass
class FineTuneArguments:
    dataset_path: str = field(default="data/qa")
    model_path: str = field(default="THUDM/chatglm-6b")
    model_revision: str = field(default=None)
    lora_rank: int = field(default=8)
    max_seq_length: int = field(default=1024)
    save_resume_from_checkpoint: str = field(default=None)
    device_map: str=field(default=None)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

def data_collator(tokenizer, features: list, to_device = None) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    if to_device:
        input_ids = input_ids.to(to_device)
        labels = labels.to(to_device)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

def load_train_model(model_path, lora_rank, model_revision: str = None, cache_dir: str = None, device_map: str="auto"):
    # init model
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, device_map=device_map, revision=model_revision, cache_dir = cache_dir
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    return model


def main():
    writer = SummaryWriter()
    fine_tune_args, training_args = HfArgumentParser(
        (FineTuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    
    import wandb
    project_name = "luotuo-qa-b-v1"
    wandb.init(project=project_name)
    
    tokenizer = AutoTokenizer.from_pretrained(fine_tune_args.model_path, trust_remote_code=True, revision=fine_tune_args.model_revision)
    model = load_train_model(fine_tune_args.model_path, fine_tune_args.lora_rank, model_revision=fine_tune_args.model_revision, device_map=fine_tune_args.device_map)

    # load dataset
    # dataset = datasets.load_from_disk(fine_tune_args.dataset_path)
    from datasets import load_dataset

    raw_datasets = load_dataset("Logic123456789/Luotuo-QA-B")
    dataset = raw_datasets
    # filter language is "Chinese"
    dataset = dataset.filter(lambda example: example["language"] == "Chinese")
    dataset = dataset["train"].train_test_split(train_size=0.9, seed=42)
    import transformers

    skip_overlength = False
    config = transformers.AutoConfig.from_pretrained(fine_tune_args.model_path, trust_remote_code=True, device_map=fine_tune_args.device_map, revision=fine_tune_args.model_revision)
    def tokenize_and_split(examples):
        result = {
            "input_ids": [],
            "seq_len": [],
        }
        for i, story in enumerate(examples["story"]):
            for j, (question, answer) in enumerate(zip(examples["questions"][i], examples["answers"][i])):
                # print(i, story, question, answer)
                context, origin_question, target = get_context_item(story, question, answer)
                if context is not None:
                    feature = preprocess(tokenizer, config, context, target, fine_tune_args.max_seq_length)
                    if skip_overlength and len(feature["input_ids"]) > fine_tune_args.max_seq_length:
                        continue
                    feature["input_ids"] = feature["input_ids"][:fine_tune_args.max_seq_length]
                    result["input_ids"].append(feature["input_ids"])
                    result["seq_len"].append(feature["seq_len"])
        return result
    dataset = dataset.map(tokenize_and_split, batched=True, num_proc=16, remove_columns=dataset['train'].column_names)
    print(f"dataset: { dataset }\n")

    # start train
    if fine_tune_args.save_resume_from_checkpoint is not None and fine_tune_args.save_resume_from_checkpoint != "":
        training_args.max_steps = 1
        training_args.output_dir = fine_tune_args.save_resume_from_checkpoint
        training_args.resume_from_checkpoint = fine_tune_args.save_resume_from_checkpoint
    def inner_data_collator(features: list) -> dict:
        return data_collator(tokenizer, features)

    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_args,
        data_collator=inner_data_collator,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
