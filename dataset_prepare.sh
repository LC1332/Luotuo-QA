set -e

dataset_name=qa-cn

python dataset_cover2jsonl.py --data_path /media/nas/ai/dataset/luotuo/${dataset_name}.json --save_path data/${dataset_name}-origin.jsonl
python dataset_jsonl_clean.py --data_path data/${dataset_name}-origin.jsonl --save_path data/${dataset_name}.jsonl

python dataset_tokenize_rows.py \
    --jsonl_path data/${dataset_name}.jsonl \
    --save_path data/${dataset_name} \
    --max_seq_length 1024 \
    --skip_overlength 1024 \
    --model_name "THUDM/chatglm-6b" \
    --num_proc 16

head -n 10000 data/${dataset_name}.jsonl >data/${dataset_name}-10000.jsonl
python dataset_tokenize_rows.py \
    --jsonl_path data/${dataset_name}-10000.jsonl \
    --save_path data/${dataset_name}-10000 \
    --max_seq_length 1024 \
    --skip_overlength 1024 \
    --model_name "THUDM/chatglm-6b" \
    --num_proc 16

tail -n 1000 data/${dataset_name}.jsonl >data/${dataset_name}-tail-1000.jsonl
python dataset_tokenize_rows.py \
    --jsonl_path data/${dataset_name}-tail-1000.jsonl \
    --save_path data/${dataset_name}-tail-1000 \
    --max_seq_length 1024 \
    --skip_overlength 1024 \
    --model_name "THUDM/chatglm-6b" \
    --num_proc 16
