# 构建训练、测试集输入
python3 combine_train.py
python3 make_standard_name_set.py # 训练、测试
#python3 make_api_set.py # 训练、测试
#ym
python3 ./ym/query_choose_api.py
python3 ./ym/test_choose_api.py
python3 ./ym/test_get_final_choose.py
# 训练
#sh finetune_api_lora.sh
sh finetune_standard_name.sh
# 推理
python3 predict_batch_standard_name.py
#python3 predict_batch_api.py

#yzh
python3 ./yzh/query_choose_api.py
python3 ./yzh/test_choose_api.py
python3 ./yzh/test_get_final_choose.py
# 训练
sh ./yzh/step3_train.sh
#推理
python3 predict_batch_standard_name.py



