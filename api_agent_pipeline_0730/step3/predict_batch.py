from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
from tqdm import tqdm
from collections import Counter
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training, LoraConfig

tokenizer = AutoTokenizer.from_pretrained("../../../../fast-data/Qwen2-7B-Instruct", trust_remote_code=True)
tokenizer.padding_side="left"
model = AutoModelForCausalLM.from_pretrained(
    "../../../../fast-data/Qwen2-7B-Instruct",
    device_map="cuda:1",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, "./output_qwen_7b_lora_standard_name")
model = model.merge_and_unload()
model = model.eval()

def find_mode(d_array):
    # 直接统计每个子列表的出现次数
    count = Counter(d_array)

    # 找出出现次数最多的子列表
    max_count = max(count.values())
    mode_list = [sublist for sublist, count in count.items() if count == max_count]

    return mode_list

def get_Res(texts, voting_times=1):
    prompt_texts = []
    for text in texts:
        messages = [
            {"role": "user", "content": text}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_texts.append(text)
    model_inputs = tokenizer(prompt_texts, padding=True, return_tensors="pt").to("cuda:1")
    do_sample = False
    if voting_times > 1:
        do_sample = True
    responses = []
    for vote in range(voting_times):
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=64
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    if voting_times == 1:
        return responses[0]
    response = []
    for i in range(len(responses[0])):
        res = str(find_mode([r[i] for r in responses])[0])
        #for r in responses:
        #    if (len(str(r[i]).split('，')) > len(res.split('，'))) or (res.strip()==""):
        #        res = str(r[i])
        response.append(res)
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
#origin
#f = open("step3_choose_keywords_input_test.txt",'r',encoding='utf-8')
#fw = open("step3_test_result_7b_withdev_voting.jsonl",'w',encoding='utf-8')

# 4096 version
#f = open("../test_top20_keywords.json",'r',encoding='utf-8')
#fw = open("step4_input_4096_keywords.txt",'w',encoding='utf-8')

# 2048 version
#f = open("../test_top20_keywords_2048.json",'r',encoding='utf-8')
#fw = open("step4_input_2048_keywords.txt",'w',encoding='utf-8')

# test_b
#f = open("../b/test_b_standard_names.json",'r',encoding='utf-8')
#fw = open("../b/test_b_real_standard_names.txt",'w',encoding='utf-8')
f = open("../step1/test_b_standard_names.json",'r',encoding='utf-8')
fw = open("../data/standard_name_test_b.txt",'w',encoding='utf-8')

batch_idx = 0
batch_size = 16
inputs = []
for line in tqdm(f):
    batch_idx += 1
    #line = line.strip().split('\001')
    #input_ = line[0]
    line = json.loads(line.strip())
    input_ = line['input']
    inputs.append(input_)
    if batch_idx >= batch_size:
        # 10选1
        labels = get_Res(inputs, 10)
        inputs = []
        batch_idx = 0
        for label in labels:
            fw.write(label+'\n')
    # json_label = json.loads(label)

if len(inputs) > 0:
    labels = get_Res(inputs,3)
    for label in labels:
        fw.write(label+'\n')
# for lines in tqdm(f):
#     line = json.loads(lines)
#     input_ = line.get("input")
#     query = get_query(input_)
#     label = get_Res(input_)
#     print({"query":query,"choose_api":label})
#     step3_query_to_choose.append({"query":query,"choose_api":label})
# json.dump(step3_query_to_choose,fw,ensure_ascii=False)






