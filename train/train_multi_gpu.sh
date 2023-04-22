set -e

# rm -rf runs
# rm -rf output/runs

dataset_name=qa-cn

deepspeed --num_gpus 2 $(dirname "$0")/train.py \
    --dataset_path $(dirname "$0")/../data/${dataset_name} \
    --lora_rank 8 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_steps 50 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --save_total_limit 4 \
    --output_dir $(dirname "$0")/../output \
    --model_revision "969290547e761b20fdb96b0602b4fd8d863bbb85" \
    --deepspeed $(dirname "$0")/ds_config_zero3.json

sh $(dirname "$0")/infer_checkpoint.sh $(dirname "$0")/..output/
