set -e

# rm -rf runs
# rm -rf output/runs

python train.py \
    --dataset_path data/qa \
    --lora_rank 16 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_steps 20 \
    --learning_rate 1e-4 \
    --fp16 \
    --device_map "auto" \
    --remove_unused_columns false \
    --logging_steps 50 \
    --save_total_limit 4 \
    --output_dir output \
    --model_revision "969290547e761b20fdb96b0602b4fd8d863bbb85"
