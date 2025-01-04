export CUDA_VISIBLE_DEVICES=0

python3 run_finetune_lora_apis.py \
    --model_name_or_path ../../../../fast-data/Qwen2-7B-Instruct \
    --data_path ./step3_LLM_train.json \
    --output_dir output_qwen2_7b_lora_api_withdev \
    --num_train_epochs 6 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --report_to "none" \
    --model_max_length 1600 \
    --lazy_preprocess True \
    --use_lora True \
    --bf16 True \
    --gradient_checkpointing
