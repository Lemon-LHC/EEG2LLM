CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path /data/lhc/models/Llama-3.2-1B-Instruct \
    --adapter_name_or_path /data/lhc/saves/Llama-3.2-1B-Instruct/lora/edf197_100hz_10000ms_tok8521_train/ \
    --template alpaca \
    --finetuning_type lora \
    --export_dir /data/lhc/models_new/llama_edf197_10000_balanced \
    --export_size 1000 \
    --export_device auto \
    --export_legacy_format False