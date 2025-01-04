import sys
import json
def convert(input_):
    prompt = "api0名称：xx，所需参数：，依赖api：；apin名称：xx，所需参数：，依赖api：；"
    json_data = json.loads(input_['output'])
    res = ""
    idx = 0
    for api in json_data['relevant APIs']:
        prompt = f"api{str(api['api_id'])}名称："
        prompt += "{api_name}，工具名：{tool_name}，所需参数：{required_parameters}，依赖api：{rely_apis}；"
        prompt = prompt.replace("{api_name}",str(api['api_name']))
        prompt = prompt.replace("{required_parameters}",str(api['required_parameters']))
        prompt = prompt.replace("{rely_apis}",str(api['rely_apis']))
        prompt = prompt.replace("{tool_name}",str(api['tool_name']))
        res += prompt
        idx += 1
    res = res.strip("；") + "。"
    res += f"最终结果：{json_data['result']}。"
    return {"input":input_['input'], "output":res}

if __name__ == "__main__":
    for line in sys.sdin:
        print(json.dumps(convert(json.loads(line), ensure_ascii=False))
    
