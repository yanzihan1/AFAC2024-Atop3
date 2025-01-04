from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
import torch
from tqdm import tqdm
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training, LoraConfig

tokenizer = AutoTokenizer.from_pretrained("../../../../fast-data/Qwen2-7B-Instruct", trust_remote_code=True)
tokenizer.padding_side="left"
model = AutoModelForCausalLM.from_pretrained(
    "../../../../fast-data/Qwen2-7B-Instruct",
    device_map="cuda:1",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(model, "../step3/output_qwen2_7b_lora_api_withdev")
model = model.merge_and_unload()
model = model.eval()

def get_Res(texts):
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

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        num_beams=5,
        num_return_sequences=5,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids
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
f = open("../step3/train_withdev.json",'r',encoding='utf-8')
fw = open("api_choose_beam_search_result_withdev.txt",'w',encoding='utf-8')

batch_idx = 0
batch_size = 6
inputs = []
for line in tqdm(f):
    batch_idx += 1
    line = json.loads(line)
    input_ = line['input']
    inputs.append(input_)
    if batch_idx >= batch_size:
        labels = get_Res(inputs)
        if len(labels) == len(inputs) * 5:
            for idx,label in enumerate(labels):
                fw.write(get_query(inputs[idx // 5])+'\001'+label+'\n')
            fw.flush()
                #fw.write(get_query(inputs[idx // 5]) + '\001' + label+'\n')
        inputs = []
        batch_idx = 0
    # json_label = json.loads(label)

if len(inputs) > 0:
    labels = get_Res(inputs)
    if len(labels) == len(inputs) * 5:
        for idx,label in enumerate(labels):
            fw.write(get_query(inputs[idx // 5])+'\001'+label+'\n')
            #fw.write(get_query(inputs[idx // 5]) + '\001' + label+'\n')
# for lines in tqdm(f):
#     line = json.loads(lines)
#     input_ = line.get("input")
#     query = get_query(input_)
#     label = get_Res(input_)
#     print({"query":query,"choose_api":label})
#     step3_query_to_choose.append({"query":query,"choose_api":label})
# json.dump(step3_query_to_choose,fw,ensure_ascii=False)






