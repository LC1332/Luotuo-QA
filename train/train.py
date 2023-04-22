from transformers.integrations import TensorBoardCallback
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

@dataclass
class FineTuneArguments:
    dataset_path: str = field(default="data/qa")
    model_path: str = field(default="THUDM/chatglm-6b")
    model_revision: str = field(default=None)
    lora_rank: int = field(default=8)
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

    tokenizer = AutoTokenizer.from_pretrained(fine_tune_args.model_path, trust_remote_code=True, revision=fine_tune_args.model_revision)
    model = load_train_model(fine_tune_args.model_path, fine_tune_args.lora_rank, model_revision=fine_tune_args.model_revision, device_map=fine_tune_args.device_map)

    # load dataset
    dataset = datasets.load_from_disk(fine_tune_args.dataset_path)
    print(f"dataset len: {len(dataset)=}\n")

    # start train
    if fine_tune_args.save_resume_from_checkpoint is not None and fine_tune_args.save_resume_from_checkpoint != "":
        training_args.max_steps = 1
        training_args.output_dir = fine_tune_args.save_resume_from_checkpoint
        training_args.resume_from_checkpoint = fine_tune_args.save_resume_from_checkpoint
    def inner_data_collator(features: list) -> dict:
        return data_collator(tokenizer, features)

    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=inner_data_collator,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
