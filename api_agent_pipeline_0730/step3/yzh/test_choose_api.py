# coding=utf8
import pandas as pd
import json
from tqdm import tqdm
from config import config
from recall_api import recall_api_func

# test  step3 LLM只选择api

class CreateTrainData:
    def __init__(self) -> None:
        self.agent_tool = config()

    def load_standard_name(self):
        data_stock = pd.read_excel(self.agent_tool.train_xlsx, sheet_name='股票标准名')
        data_fund = pd.read_excel(self.agent_tool.train_xlsx, sheet_name='基金标准名')
        standard_name = data_stock['标准股票名称'].to_list() + data_fund['标准基金名称'].to_list()
        return standard_name

    def df_2_json(self, df, json_file):
        print('正在处理', json_file)
        with open(json_file, 'w') as m:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                query = row['query']
                query = query.replace("健盛\n","健盛")
                # recall api
                recall_api_string = recall_api_func(query)
                input_ = "你现在是一个金融领域专家，你需要深入理解用户query和可能需要的api，选择最终需要的api，尽可能按照调用顺序进行排序。" \
                         "query是：{}。" \
                         "可能需要的api：{}。".format(query, recall_api_string)
                single_data = {'input': input_}
                m.write(json.dumps(single_data, ensure_ascii=False) + '\n')

    def run(self):
        df_test_a = pd.read_excel(self.agent_tool.test_b_data_xlsx)
        self.df_2_json(df_test_a, self.agent_tool.test_step3_save_path)


p = CreateTrainData()
p.run()