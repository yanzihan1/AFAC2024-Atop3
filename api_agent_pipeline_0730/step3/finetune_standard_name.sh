export CUDA_VISIBLE_DEVICES=0

python3 run_finetune_standard_name.py \
    --model_name_or_path ../../../../fast-data/Qwen2-7B-Instruct \
    --data_path ../data/transet_standard_name.json \
    --output_dir output_qwen_7b_lora_standard_name \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --report_to "none" \
    --model_max_length 768 \
    --lazy_preprocess True \
    --use_lora True \
    --bf16 True \
    --gradient_checkpointing
