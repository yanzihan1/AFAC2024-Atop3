import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import difflib
import pandas as pd
from config import config

myconfig = config()
tokenizer = AutoTokenizer.from_pretrained(myconfig.recall_keywords_reranker_model_path)
model = AutoModelForSequenceClassification.from_pretrained(myconfig.recall_keywords_reranker_model_path)
model.to("cuda:2")
model.eval()
def load_standard_name():
    data_stock = pd.read_excel(myconfig.train_xlsx, sheet_name='股票标准名')
    data_fund = pd.read_excel(myconfig.train_xlsx, sheet_name='基金标准名')
    standard_name = data_stock['标准股票名称'].to_list() + data_fund['标准基金名称'].to_list()
    return standard_name
standard_name_list = load_standard_name()
api_path = open(myconfig.api_path,'r',encoding='utf-8')
api_dict = json.load(api_path)
tool_api2description = {}

def recall_keywords_func(query):
    with torch.no_grad():
        ans2score={}
        products = difflib.get_close_matches(query, standard_name_list, n=2048, cutoff=0.0001)
        pairs = [[query, ans] for ans in products]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda:2")
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        for i, j in zip(pairs, scores):
            ans = i[1]
            score = j
            ans2score[ans] = score
        top_20_items = sorted(ans2score.items(), key=lambda item: item[1], reverse=True)
        top_15_items_list = [i[0] for i in top_20_items][:15]
        return top_15_items_list







