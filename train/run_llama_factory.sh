#!/bin/bash
# 运行LLaMA-Factory分布式训练

cd /data/lhc/projects/fine
python train.py \
    --model_name /data/lhc/models/Llama-3.2-1B-Instruct \
    --dataset_dir /data/lhc/datasets_new/sleep \
    --train_dataset edf5_100hz_10000ms_tok8363_train \
    --test_dataset edf5_100hz_10000ms_tok8363_test \
    --llamafactory_path /data/lhc/projects/LLaMA-Factory \
    --output_dir /data/lhc/saves/Llama-3.2-1B-Instruct/lora/edf5_100hz_10000ms_tok8363_train \
    --gpu_ids 0,1,2,3 \
    --learning_rate 5e-05 \
    --num_epochs 6.0 \
    --cutoff_len 500 \
    --train_batch_size 1 \
    --grad_accum_steps 4 \
    --warmup_steps 50 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --val_size 0.11 \
    --save_steps 20 \
    --logging_steps 20 \
    --test_interval 20 \
    --eval_precision fp16 \
    --eval_while_training 