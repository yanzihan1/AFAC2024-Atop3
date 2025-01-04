#训练





#推理

#================有关api的部分
# 得到训练集和验证集的 api 召回top20 以及 选择的label
python query_choose_api.py
# 得到测试集的top20api
python test_choose_api.py
# 根据测试集的top20api 得到最终选择的api
python test_get_final_choose.py

#================有关keywords的部分
#训练+验证+测试的
python get_keywords.py
