# 这个py
# coding=utf8
import random

import pandas as pd
import difflib
import json
from tqdm import tqdm

#  制造reranker 召回api关键词



class CreateTrainData:
    def __init__(self) -> None:
        self.standard_name = self.load_standard_name()
        self.train_recall_api_keywords = open("train_keywords.txt",'w',encoding='utf-8')
        self.dev_recall_api_keywords = open("dev_keywords.txt",'w',encoding='utf-8')
        self.candicate_num = 10 # 计算jaccard的标准名称候选集大小

    def get_jaccard(self,a,b):
        lis_a = [a[i] for i in range(len(a))]
        lis_b = [b[i] for i in range(len(b))]
        return len(list(set(a)&set(b))) / len(set((set(a)|set(b))))

    def load_standard_name(self):
        data_stock = pd.read_csv("../data/standard_stock.txt", sep='\001',header=None)
        data_fund = pd.read_csv("../data/standard_fund.txt", sep='\001',header=None)
        standard_name = data_stock[0].to_list() + data_fund[0].to_list()
        return standard_name

    def df_2_json(self, df, recall_api_keywords):
        cnt = 0
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            query = row[0]
            label = row[1]
            cur_standard_name = []
            relevant_APIs = json.loads(label)
            relevant_APIs_ = relevant_APIs.get("relevant APIs")
            for line in relevant_APIs_:
                required_parameters = line.get("required_parameters")
                for required_parameter in required_parameters:
                    if required_parameter in self.standard_name or required_parameter[0] in self.standard_name:
                        cur_standard_name.append(required_parameter)
            leng = len(cur_standard_name)
            if leng>=1:
                cnt += 1
                neg = []
                pos = [i[0] for i in cur_standard_name]
                # 挑选负样本
                for pos_item in pos:
                    idx = 0
                    random.shuffle(self.standard_name)
                    for neg_item in self.standard_name:
                        if idx >= self.candicate_num:
                            break
                        if neg_item not in pos:
                            jaccard = 0
                            jaccard = self.get_jaccard(pos_item,neg_item)
                            neg.append([neg_item,jaccard])
                        idx += 1
                    neg_sample = sorted(neg, key=lambda x: x[1], reverse=True)[0][0]
                    line = '\001'.join([query.strip(),pos_item,neg_sample])
                    recall_api_keywords.write(line+'\n')
        return cnt

    def run(self):
        df_train = pd.read_excel('../data/train.xlsx')
        df_dev = pd.read_excel('../data/dev.xlsx')
        # df_test_a = pd.read_excel('/mnt/mcu/yanzihan/agent-code/data/data-0520/test_a.xlsx')

        cnt = self.df_2_json(df_train, self.train_recall_api_keywords)
        print(cnt)
        cnt = self.df_2_json(df_dev, self.dev_recall_api_keywords)
        print(cnt)
        # self.df_2_json(df_test_a)
        # print(self.res)

p = CreateTrainData()
p.run()
