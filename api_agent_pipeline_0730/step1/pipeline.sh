# Step1：使用Reranker模型（Top 15）及LLM表征（Top 5），召回候选标准名
# 构造训练数据
cat ../data/standard_fund.txt ../data/standard_stock.txt > ../data/all_standard_name.txt
python3 my_query_reranker.py
cat train_keywords.txt dev_keywords.txt | python3 get_train_data.py > train_withdev.txt
# 训练LLM表征模型
sh train.sh
awk -F '\001' '{print "query："$1"。请生成query中涉及的基金或股票实体的表征："}' ../data/test_b.txt > ./prompt_test_queries.txt
# 推理得到Top 5 候选标准名
sh predict.sh
sh find_k_origin.sh
cat test_result_withdev.txt | python3 process_data.py test_result_withdev
python3 get_train_queries.py | awk -F '\001' '{print "query："$1"。请生成query中涉及的基金或股票实体的表征："}' > ./prompt_train_queries.txt
sh find_k.sh 
cat train_result_withdev.txt | python3 process_data.py train_result_withdev
# 与Reranker的Top15 keywords去重后混合，得到train、dev、test的分别的Top 20 候选标准名
#==============
  缺少 yzh 训练部分 训练step3model
#==============
  缺少 yzh 推理部分 得到step3_test.json 存在目录data下
#==============
# 训练集，用于step3的训练
python3 combine.py reranker_train_withdev_15.txt train_result_withdev_15.txt train_pred_standard_names_withdev.txt
# 测试集，用于step3的推理得到step4的最终输入标准名
python3 combine.py reranker_test_withdev_15.txt test_result_withdev_15.txt test_pred_standard_names.txt
