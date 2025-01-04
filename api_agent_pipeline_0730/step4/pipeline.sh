# 获取格式1的step4 train 输入
# 获取扰动源数据（step3中api模型beam search数据）
python3 predict_batch_api.py
cat xx | awk -F '\001' 'NF==2{print}' > beam_search_api_result.txt
# step4训练集的base为无扰动数据
python3 get_train_info.py
python3 make_step4_input.py train_api.json train_standard_name.json step4_best_format_train_input.json train
cat trainset_base.json > trainset_base.json
cat trainset_base.json >> trainset_base.json
cat trainset_base.json >> trainset_base.json
cat trainset_base.json >> trainset_base.json
cat trainset_base.json >> trainset_base.json

cp trainset_base.json trainset_ratio1.json
cp trainset_base.json trainset_ratio2.json
python3 add_noise.py beam_search_api_result.txt 1800 4400 2800 >> trainset_ratio1.json
python3 add_noise.py beam_search_api_result.txt 1500 3667 2333 >> trainset_ratio2.json
cat trainset_ratio1.json | shuf | python3 change_percent.py | python3 make_no_json_res.py > x.txt
mv x.txt trainset_ratio1.json
cat trainset_ratio2.json | shuf | python3 change_percent.py | python3 make_no_json_res.py > x.txt
mv x.txt trainset_ratio2.json
# 获取格式1的step4 test_b输入
cat ../data/test_b.txt | python3 change_percent.py > test_b_queries.txt
paste test_b_queries.txt ../data/standard_name_test_b.txt | python3 make_json_data.py > standard_name_with_queries.txt
python3 make_step4_input.py b_step3_pro.json test_standard_name.txt step4_best_format_test_input.json test

# 获取without step3的train输入

# 获取without step3的test_b输入


# 获取增强query
# @锦玲

# 训练
# @锦玲
# 推理
# @锦玲

# 投票
# @锦玲
# 后处理
# 对投票结果进行后处理
sh post_process.sh voting.jsonl _voting.jsonl
# join 训练集
python3 join_train.py _voting.jsonl submit_result.jsonl

