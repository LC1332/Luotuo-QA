set -e

python app/webui.py --share true --server_port 12345 --peft_path ./output --with_origin_model true
