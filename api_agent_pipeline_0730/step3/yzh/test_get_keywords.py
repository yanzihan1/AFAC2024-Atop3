import pandas as pd
import json
from tqdm import tqdm
from test_recall_keywords import recall_keywords_func
from config import config

class CreateTrainData:
    def __init__(self) -> None:
        self.agent_tool = config()
    def load_standard_name(self):
        data_stock = pd.read_excel(self.agent_tool.train_xlsx, sheet_name='股票标准名')
        data_fund = pd.read_excel(self.agent_tool.train_xlsx, sheet_name='基金标准名')
        standard_name = data_stock['标准股票名称'].to_list() + data_fund['标准基金名称'].to_list()
        return standard_name

    def get_choose_api(self,file_):
        f = open(file_,'r',encoding='utf-8')  #这里是step3 LLM 推理得出的结果
        query2api = {}
        for line in f:
            line = json.loads(line)
            query = line.get('query')
            choose_api = line.get('choose_api')
            query2api[query] = choose_api
        return query2api

    def df_2_json(self, df, json_file,dev=False):
        self.standard_name = self.load_standard_name()
        with open(json_file, 'w') as m:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                query = row['query']
                query = query.replace("健盛\n","健盛")
                recall_keywords_list = recall_keywords_func(query)
                keywords_top15_string = "，".join(recall_keywords_list)
                input_ = "query是：{}。" \
                         "query中提到的产品标准名可能是：{}。".format(query,keywords_top15_string)
                if dev:
                    label = row['label']
                    required_parameter_label = []
                    relevant_APIs = json.loads(label)
                    relevant_APIs_ = relevant_APIs.get("relevant APIs")
                    for line in relevant_APIs_:
                        required_parameters = line.get("required_parameters")
                        for required_parameter in required_parameters:
                            if required_parameter in self.standard_name or required_parameter[0] in self.standard_name:
                                required_parameter_label.append(required_parameter)
                    single_data = {'input': input_, "output": label}
                else:
                    single_data = {'input': input_}
                m.write(json.dumps(single_data, ensure_ascii=False) + '\n')

    def run(self):

        df_test_a = pd.read_excel(self.agent_tool.train_xlsx)
        self.df_2_json(df_test_a, self.agent_tool.train_keywords)

        df_test_a = pd.read_excel(self.agent_tool.dev_data_xlsx)
        self.df_2_json(df_test_a, self.agent_tool.dev_keywords)

        df_test_a = pd.read_excel(self.agent_tool.test_b_data_xlsx)
        self.df_2_json(df_test_a, self.agent_tool.test_keywords)


p = CreateTrainData()
p.run()