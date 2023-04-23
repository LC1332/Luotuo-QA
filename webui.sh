set -e

python app/webui.py --share true --server_port 12345 --peft_path "silk-road/luotuo-qa-lora-0.1" --with_origin_model true
