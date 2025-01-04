import sys
import json
import pandas as pd

prompt = "你现在是一个金融领域专家，你需要深入理解用户query的意图，并结合金融知识，选择最终用户实际需要的产品标准名。 \n query：{QUERY}。 \n 可能的产品标准名：{STANDARD_NAMES}。请选择："

def get_query_and_labels(path):
    all_standard_names = pd.concat([pd.read_csv('../data/standard_name_stock.txt',sep='\001',header=None,index=False), pd.read_csv('../data/standard_name_fund.txt',sep='\001',header=None,index=False)])
    all_standard_names = set(all_standard_names[0].tolist())
    labels = []
    queries = []
    if 'xlsx' in path:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep='\001',header=None,quoting=3)
    for idx in range(df.shape[0]):
        standard_names = []
        queries.append(df.iloc[i, 0])
        if df.shape[1] > 1:
            json_str = df.iloc[i,1]
            j = json.loads(json_str)
            for api in j['relevant APIs']:
                if type(api['required_parameters']) == list and type(api['required_parameters'])[0]) == list:
                    if api['required_parameters'])[0][0] in all_standard_names:
                        standard_names.append(api['required_parameters'])[0][0])
            labels.append('，'.join(standard_names))
    if df.shape[1] == 1:
        return queries
    return queries, labels

def get_pred_standard_names(path):
    preds = []
    with open(path,'r') as f:
        for line in f:
            preds.append(line.strip())
    return preds

if __name__ == "__main__":
    train_query, train_labels = get_query_and_labels('../data/train_withdev.xlsx')
    train_preds = get_pred_standard_names('../step1/train_pred_standard_names_withdev.txt')
    test_queries = get_query_and_labels('../data/test_b.txt')
    test_preds = get_pred_standard_names('../step1/test_pred_standard_names.txt')
    fw_train = open('trainset.json','w')
    fw_test = open('testset.json','w')
    for idx,query in enumerate(train_queries):
        res = {"input": prompt.replace('{QUERY}',query).replace('{STANDARD_NAMES}',train_preds[idx]), "output":train_labels[idx]}
        fw_train.write(json.dumps(res, ensure_ascii=False)+'\n')

    for idx,query in enumerate(test_queries):
        res = {"input": prompt.replace('{QUERY}',query).replace('{STANDARD_NAMES}',test_preds[idx])}
        fw_test.write(json.dumps(res, ensure_ascii=False)+'\n')
