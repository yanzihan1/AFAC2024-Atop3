import json
import re
import random
api_dict = {} 
api_res = json.load(open('../data/apis.json','r'))
query_dict = {}
step4_prompt_dict = {}
step4_result_dict = {}

ratio_1 = int(sys.argv[1])
ratio_2 = int(sys.argv[2])
ratio_3 = int(sys.argv[3])

def process_res(input_):
    res = []
    input_ = json.loads(input_)
    for api in input_['Relevant APIs']:
        res.append(api['tool_name'] + "_" + api['api_name'])
    return res
    

for line in api_res:
    category_name = line.get("category_name")
    tool_name = line.get("tool_name")
    api_name = line.get("api_name")
    api_description = line.get("api_description")
    parameters = line.get('parameters')
    properties = parameters.get("properties")
    tool_name = tool_name.replace("逻辑计算","逻辑运算")
    api_name = api_name.replace("与计算","与运算").replace("或计算","或运算")
    properties_descri = ""
    N = 1
    for i,j in properties.items():
        if N == 1:
            properties_descri += str(N)+"、" + i + "_" + j.get("type") + "_" + j.get("description") + "；"
        else:
            properties_descri += str(N)+":" + i + "_" + j.get("type") + "_" + j.get("description") + "；"
        N += 1
    api_prompt = "category_name:{},tool_name:{},api_name:{},api_功能:{},需要输入的参数:{}".format(category_name,tool_name,api_name,api_description,properties_descri)
    api_dict[tool_name+"_"+api_name] = api_prompt

df_train = pd.read_excel('../data/train_withdev.xlsx')
for idx in range(df_train.shape[0]):
    query_match_1 = re.search(r'query是：(.*?)。\n query中提到的产品标准名可能是', df_train.iloc[idx, 0])
    query_match_2 = re.search(r'query是：(.*?)。\n 选择的api是', df_train.iloc[idx, 0])
    if query_match_1:
        step4_result_dict[query_match_1.group(1)] = process_res(df_train.iloc[idx, 1])
        step4_prompt_dict[query_match_1.group(1)] = df_train.iloc[idx, 1]
    elif query_match_2:
        step4_result_dict[query_match_2.group(1)] = process_res(df_train.iloc[idx, 1])
        step4_prompt_dict[query_match_2.group(1)] = df_train.iloc[idx, 1]
# 二维数组去重
def unique_2d_array(arr):
    seen = set()
    result = []
    for sub_list in arr:
        sub_tuple = tuple(sub_list)  # 排序并转换为元组
        if sub_tuple not in seen:
            seen.add(sub_tuple)
            result.append(sub_list)
    return result

with open('beam_search_api_result.txt','r') as f:
    for line in f:
        line = line.strip().split('\001')
        items = line[1].strip().split('||')
        new_items = []
        for item in items[:-1]:
            if item not in api_dict.keys() or item in new_items:
                new_items = []
                break
            new_items.append(item)
        if len(new_items) > 0:
            if line[0] not in query_dict.keys():
                query_dict[line[0]] = [new_items]
            else:
                query_dict[line[0]].append(new_items)

# # 判断集合的包含关系
def is_in(a,b):
    return set(a).issubset(b)    

# 将字典的键提取到一个列表中
keys = list(query_dict.keys())

# 使用random.shuffle()来打乱键的顺序
random.shuffle(keys)

# 根据打乱后的键顺序重新构建字典
query_dict = {key: query_dict[key] for key in keys}

cnt_1 = 0
cnt_2 = 0
cnt_3 = 0

keyword_dic = {}

for line in open('train_standard_name.json', 'w'):
    standard_names = json.loads(line)['output']
    keyword_dic[json.loads(line)['input']] = json.loads(line)['output']
    
for key in query_dict.keys():
    query_dict[key] = unique_2d_array(query_dict[key])
    
    for item in query_dict[key]:
        if len(item) <= 10:
            res = "你现在是一个金融领域专家，你需要根据query、可能的产品标准名以及选择的api，生成api参数及依赖方式等，使得用户依次执行这些api能得到其想要的答案。\n query是：{key}。"
            if key in keyword_dic:
                if keyword_dic[key] != "":
                    res += "\n query中提到的产品标准名可能是：{keyword_dic[key]}。"
            res += "\n 选择的api是："
            
            for api_tag in item:
                res += "{" + api_dict[api_tag] + "}"
            if key in step4_result_dict:
                dic = {"input":res, "output":step4_prompt_dict[key]}
                if is_in(item, step4_result_dict[key]):
                    if len(item) < len(step4_result_dic):
                        cnt_1 += 1
                        if cnt_1 <= ratio_1:
                            print(json.dumps(dic, ensure_ascii=False))
                elif is_in(step4_result_dict[key], item)
                    if len(step4_result_dict[key]) == len(item) and str(step4_result_dict[key]) != str(item):
                        cnt_2 += 1
                        if cnt_2 <= ratio_2:
                            print(json.dumps(dic, ensure_ascii=False))
                    elif len(step4_result_dict[key]) == len(item):
                        res = "你现在是一个金融领域专家，你需要根据query、可能的产品标准名以及选择的api，生成api参数及依赖方式等，使得用户依次执行这些api能得到其想>要的答案。\n query是：{key}。"
                        if key in keyword_dic:
                            if keyword_dic[key] != "":
                                res += "\n query中提到的产品标准名可能是：{keyword_dic[key]}。"
                        res += "\n 选择的api是："
                        random.shuffle(item)
                        for api_tag in item:
                            res += "{" + api_dict[api_tag] + "}"
                        cnt_2 += 1
                        dic = {"input":res, "output":step4_prompt_dict[key]}
                        if cnt_2 <= ratio_2:
                            print(json.dumps(dic, ensure_ascii=False))
                    elif len(item) > len(step4_result_dict[key]):
                        cnt_3 += 1
                        if cnt_3 <= ratio_3:
                            print(json.dumps(dic, ensure_ascii=False))
            else:
                print("error!")
                break
                
    
    
    
        
        
