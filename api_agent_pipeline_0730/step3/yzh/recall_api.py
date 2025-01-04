import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import config
import json

agent_tool = config()
tokenizer = AutoTokenizer.from_pretrained(agent_tool.recall_api_reranker_model_path)
model = AutoModelForSequenceClassification.from_pretrained(agent_tool.recall_api_reranker_model_path)
device = "cuda:3"

model.to(device)
model.eval()

api_path = open(agent_tool.api_path,'r',encoding='utf-8')
api_dict = json.load(api_path)
tool_api2description = {}
api_list = []
api_res = {}
for line in api_dict:
    category_name = line.get("category_name")
    tool_name = line.get("tool_name")
    api_name = line.get("api_name")
    api_description = line.get("api_description")
    parameters = line.get('parameters')
    properties = parameters.get("properties")
    tool_name = tool_name.replace("逻辑计算","逻辑运算")
    api_name = api_name.replace("与计算","与运算").replace("或计算","或运算")
    properties_descri = ""
    api_res[tool_name+"_"+api_name] = tool_name+"_"+api_name+'_'+api_description  # step3用 api_description 做的匹配。  step4 用的 api_prompt
for line in api_dict:
    category_name = line.get("category_name")
    tool_name = line.get("tool_name")
    api_name = line.get("api_name")
    api_description = line.get("api_description")
    tool_replace = tool_name+'_'+api_name
    tool_replace = tool_replace.replace("逻辑计算_与计算","逻辑运算_与运算")
    tool_replace = tool_replace.replace("逻辑计算_或计算","逻辑运算_或运算")
    api_list.append(tool_replace+"_"+api_description)
def recall_api_func(query):
    with torch.no_grad():
        ans2score={}
        pairs = [[query, ans] for ans in api_list]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        for i, j in zip(pairs, scores):
            ans = i[1]
            score = j
            ans2score[ans] = score
        top_20_items = sorted(ans2score.items(), key=lambda item: item[1], reverse=True)
        top_20_items_list = [i[0] for i in top_20_items][:20]
        res = ""
        for lines in top_20_items_list:
            line = lines.split('_')
            tool_input = line[0]+"_"+line[1]
            res += "{" + api_res.get(tool_input) + "}"

        return res







