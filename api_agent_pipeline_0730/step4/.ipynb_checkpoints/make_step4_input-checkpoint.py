import sys
import json

def set_api_dict():
    api_res = json.load(open('../../data/apis.json','r'))
    api_dict = {}
    
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
        api_prompt = "category_name:{}。tool_name:{}。api_name:{}。api_功能:{}。需要输入的参数:{}".format(category_name,tool_name,api_name,api_description,properties_descri)
        api_dict[tool_name+"_"+api_name] = api_prompt

def get_input(f_api_path, f_keyword_path, output_path, stage = "test"):
    # 处理为最高分格式
    if stage == "train":
        outputs = pd.read_excel('../data/train_withdev.xlsx')['label'].tolist()
    f_api = open(f_api_path,'r')
    f_keyword = open(f_keyword_path,'r')
    fw = open(output_path,'w')
    idx = 0
    for api_line in f_api:
        res = "你现在是一个金融领域专家，你需要根据query、可能的产品标准名以及选择的api，生成api参数及依赖方式等，使得用户依次执行这些api能得到其想要的答案。\n "
        standard_name_line = f_keyword.readline()
        standard_names = json.loads(standard_name_line)['output']
        res += f"query是：{json.loads(standard_name_line)['input']}。"
        if standard_names != "" and len(standard_names) > 0:
            res += "\n query中提到的产品标准名可能是：" + standard_names + "。"
        apis = json.loads(api_line)['query'].strip().split('||')[:-1]
        res += "\n 选择的api是："
        for api in apis:
            res += "{" + api_dict[api] + "}"
        fw.write(json.dumps({"input":res, "output":outputs[idx]}, ensure_ascii=False)+"\n")
        idx += 1

set_api_dict()
get_input(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
