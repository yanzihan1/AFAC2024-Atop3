CUDA_VISIBLE_DEVICES=1 python3 find_nearest_k.py --model_path=../../../../fast-data/Qwen2-7B-Instruct \
       --lora_path=./output_qwen_step1_api_agent_withdev \
       --input_path=./prompt_train_queries.txt \
       --output_path=train_result_withdev.txt \
       --top_dict True \
       --top_corpus_path=../data/all_standard_name.txt \
       --top_k=100 \
       --ann_threshold=0.0 \
       --top_index_path=./withdevtop_index.idx
