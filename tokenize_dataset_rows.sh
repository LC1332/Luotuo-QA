set -e

data_name=qa

python cover2jsonl.py --data_path /media/nas/ai/dataset/luotuo/${data_name}.json --save_path data/${data_name}.jsonl

python tokenize_dataset_rows.py \
    --jsonl_path data/${data_name}.jsonl \
    --save_path data/${data_name} \
    --max_seq_length 1024 \
    --skip_overlength 1024 \
    --model_name "THUDM/chatglm-6b" \
    --num_proc 16

head -n 10000 data/${data_name}.jsonl >data/${data_name}-10000.jsonl
python tokenize_dataset_rows.py \
    --jsonl_path data/${data_name}-10000.jsonl \
    --save_path data/${data_name}-10000 \
    --max_seq_length 1024 \
    --skip_overlength 1024 \
    --model_name "THUDM/chatglm-6b" \
    --num_proc 16

tail -n 1000 data/${data_name}.jsonl >data/${data_name}-tail-1000.jsonl
python tokenize_dataset_rows.py \
    --jsonl_path data/${data_name}-tail-1000.jsonl \
    --save_path data/${data_name}-tail-1000 \
    --max_seq_length 1024 \
    --skip_overlength 1024 \
    --model_name "THUDM/chatglm-6b" \
    --num_proc 16
