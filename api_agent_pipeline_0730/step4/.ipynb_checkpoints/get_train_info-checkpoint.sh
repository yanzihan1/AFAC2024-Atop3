import json
import pandas as pd

df = pd.read_excel('../data/train_withdev.xlsx')

all_standard_names = pd.concat([pd.read_csv('../data/standard_name_stock.txt',sep='\001',header=None,index=False), pd.read_csv('../data/standard_name_fund.txt',sep='\001',header=None,index=False)])
all_standard_names = set(all_standard_names[0].tolist())
labels = []
queries = []

for idx in range(df.shape[0]):
    standard_names = []
    apis = []
    queries.append(df.iloc[i, 0])
    if df.shape[1] > 1:
        json_str = df.iloc[i,1]
        j = json.loads(json_str)
        for api in j['relevant APIs']:
            if type(api['required_parameters']) == list and type(api['required_parameters'])[0]) == list:
                if api['required_parameters'])[0][0] in all_standard_names:
                    standard_names.append(api['required_parameters'])[0][0])
        	apis += f"{tool_name}_{api_name}||"
        labels.append(['，'.join(standard_names), apis])

fw_api = open('train_api.json', 'w')
fw_standard_name = open('train_standard_name.json', 'w')
for item in labels:
    fw_api.write(json.dumps({"api":item[1]}, ensure_ascii=False) + '\n')
    fw_standard_name.write(json.dumps({"input":item[0]}, ensure_ascii=False) + '\n')
