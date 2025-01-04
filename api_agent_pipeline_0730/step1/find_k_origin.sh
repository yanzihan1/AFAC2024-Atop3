CUDA_VISIBLE_DEVICES=0 python3 find_nearest_k.py --model_path=../../../../fast-data/Qwen2-7B-Instruct \
       --lora_path=./output_qwen_step1_api_agent_withdev \
       --input_path=./prompt_test_queries.txt \
       --output_path=test_result_withdev.txt \
       --top_dict True \
       --top_corpus_path=./corpus_.emb \
       --prefix="withdev" \
       --ann_threshold 0.0 \
       --top_k=100
