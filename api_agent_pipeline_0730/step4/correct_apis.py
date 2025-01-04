import re
import sys
import json
import copy

api_dict = {}
apis_json = json.load(open("../data/apis.json",'r'))
for api in apis_json:
    api_dict[api['tool_name'].replace("逻辑计算","逻辑运算") + "_" + api['api_name'].replace("或计算","或运算").replace("与计算","与运算")] = api['category_name']

def get_jaccard(a,b):
        lis_a = [a[i] for i in range(len(a))]
        lis_b = [b[i] for i in range(len(b))]
        return len(list(set(a)&set(b))) / len(set((set(a)|set(b))))

for line in sys.stdin:
    idx = 0
    res = ""
    try:
        j = json.loads(line.strip())
        new_apis = []
        for api in j["relevant APIs"]:
            key = api['tool_name'] + "_" + api['api_name']
            #print(api)
            
            real_api = key
            if key not in api_dict:
                dis = 0
                for item in api_dict:
                    #if ("股票" in line and "基金" not in line and '基金' in api_dict[item]) or ("基金" in line and "股票" not in line and '股票' in api_dict[item]):
                    #    continue
                    tmp_dis = get_jaccard(item,key)
                    if tmp_dis > dis:
                        dis = tmp_dis
                        real_api = item
            api['tool_name'] = real_api.split('_')[0]
            api['api_name'] = real_api.split('_')[1]
            new_apis.append(api)
        j["relevant APIs"] = new_apis
        print(json.dumps(j,ensure_ascii=False))

        #origin_api_string = re.search(r"选择的api是：(.*?)specialxxx",json.loads(line.strip())['input']+"specialxxx")
        #api_list = [ item for item in re.findall(r"{(.*?)}", origin_api_string.group(1)) ]
        #api_list = [re.search(r"api_功能:(.*?)。需要输入的参数",item).group(1) for item in api_list]
        #for api in api_list:
        #    res += "{" + f"候选api{idx}："
        #    res += api + "；"+api_dict[api] + "}" 
        #    idx += 1
    except:
        #print(line.strip().strip('\n'))
        continue
    #print(line.replace(origin_api_string.group(1), res).strip().strip('\n'))
