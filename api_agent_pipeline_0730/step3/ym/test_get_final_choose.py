from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import re
import json
from tqdm import tqdm
from config import config


tokenizer = AutoTokenizer.from_pretrained(config.qwen2_model, trust_remote_code=True,padding_side='left')
config = AutoConfig.from_pretrained(config.qwen2_model, trust_remote_code=True)
config.model_max_length = 3000
model = AutoModelForCausalLM.from_pretrained(
    "config.qwen2_model",
    device_map="cuda:3",
    trust_remote_code=True
).eval()

def get_Res(batch_prompt):
    batch_text=[]
    for prompt in batch_prompt:
        messages = [
            {"role": "system", "content": prompt}

        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        batch_text.append(text)
    model_inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda:3")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=3000
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return response

def get_query(query):
    pattern = r'query是：(.*?)。可能需要的api'
    match = re.search(pattern, query)
    if match:
        extracted_info = match.group(1)  # group(1) 是第一个括号内匹配的内容

        return extracted_info
    else:
        print("wrong")
step3_query_to_choose = []

f = open(config.test_step3_save_path,'r',encoding='utf-8')
fw = open(config.test_step3_final_choose_path,'w',encoding='utf-8')

input_list = []
for lines in tqdm(f):
    line = json.loads(lines)
    input_ = line.get("input")
    input_list.append(input_)

s = 0
bs = 16
e = 16
while e <= len(input_list):
    cur_label = input_list[s:e]
    label_list = get_Res(cur_label)
    for label in label_list:
        print(label)
        cur_json = json.dumps({"query": "111", "choose_api": label}, ensure_ascii=False)
        fw.write(cur_json + '\n')
    e += bs
    s += bs
    print(s)
cur_label = input_list[s:]
label_list = get_Res(cur_label)
for label in label_list:
    cur_json = json.dumps({"query":"111","choose_api":label},ensure_ascii=False)
    fw.write(cur_json+'\n')






