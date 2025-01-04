# coding=utf8
import pandas as pd
import json
from tqdm import tqdm
from recall_api import recall_api_func
from config import config

# 得到召回的top20 以及 选择api的label 用作训练

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
        fw = open(json_file,'w', encoding='utf-8')
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            query = row['query']
            label = row['label']
            label_string = ""
            label_string_list = []
            relevant_APIs = json.loads(label)
            relevant_APIs_ = relevant_APIs.get("relevant APIs")
            for line in relevant_APIs_:
                tool_name = line.get("tool_name")
                api_name = line.get("api_name")
                label_string_list.append(tool_name+"_"+api_name)

            for i in label_string_list:
                 label_string += i + "||"
            query = query.replace("健盛\n","健盛")

            # recall api
            recall_api_string = recall_api_func(query)
            input_ = "你现在是一个金融领域专家，你需要深入理解用户query和可能需要的api，选择最终需要的api，尽可能按照调用顺序进行排序。" \
                     "query是：{}。" \
                     "可能需要的api：{}。".format(query,recall_api_string)
            single_data = {'input': input_, 'output': label_string}
            fw.write(json.dumps(single_data, ensure_ascii=False) + '\n')
    def run(self):

        df_test_a = pd.read_excel(self.agent_tool.train_data_xlsx)
        self.df_2_json(df_test_a, self.agent_tool.step3_save_path)

        df_test_a = pd.read_excel(self.agent_tool.dev_data_xlsx)
        self.df_2_json(df_test_a, self.agent_tool.step3_dev_save_path)

p = CreateTrainData()
p.run()