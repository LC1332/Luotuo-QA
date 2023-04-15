CHECKPOINT=$1

if [ -z "$CHECKPOINT" ]; then
    echo "Please provide a checkpoint path"
    exit 1
fi

CHECKPOINT_CONFIG="$CHECKPOINT/adapter_config.json"

MODEL_REVISION="969290547e761b20fdb96b0602b4fd8d863bbb85"

if [ ! -f "$CHECKPOINT_CONFIG" ]; then
    echo "Checkpoint $CHECKPOINT_CONFIG does not exist, loading and saving..."
    python train.py \
        --dataset_path data/qa \
        --lora_rank 16 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --save_steps 20 \
        --learning_rate 1e-4 \
        --fp16 \
        --remove_unused_columns false \
        --logging_steps 50 \
        --save_total_limit 2 \
        --output_dir output \
        --save_resume_from_checkpoint $CHECKPOINT \
        --model_revision $MODEL_REVISION
fi

python infer.py \
    --peft_path $CHECKPOINT \
    --model_revision $MODEL_REVISION
