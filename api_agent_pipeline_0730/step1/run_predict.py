import os
from dataclasses import dataclass, field
from torch.utils.data import (DataLoader, Dataset)
from typing import Dict
from utils import Qwen2ForANN
import platform
import signal
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import readline
import torch
import sys
import transformers
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training, LoraConfig
#import deepspeed

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess_myself(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
):
    """Preprocesses the data for supervised fine-tuning."""

    input_ids = []
    attention_mask = []

    for i, msg in enumerate(messages):
        #print(msg['text'])
        #print(msg['label'])
        texts = msg.strip()
        m =  [
            {"role": "user", "content": texts}
        ]
        texts = tokenizer.apply_chat_template(
            m,
            tokenize=False,
            add_generation_prompt=True
        ) 
        text = tokenizer(
                texts,
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=max_len
            )
        input_ids.append([tokenizer.pad_token_id]*(max_len - len(text['input_ids'])) + text['input_ids'])
        attention_mask.append([0]* (max_len - len(text['input_ids'])) + text['attention_mask'])
       
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    attention_mask  = torch.tensor(attention_mask, dtype=torch.int)                     

    return dict(
        input_ids=input_ids, 
        attention_mask=attention_mask,
    )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess_myself([self.raw_data[i]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

if __name__ == '__main__':
    set_seed(2024)
    model_path = sys.argv[1]
    lora_path = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model = Qwen2ForANN(
        model_path,
        config=config
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    infer_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # detect cuda capability
    model = model.to(infer_dtype)
    model.eval()
    model = model.cuda()
    
    dataset_cls = (
        LazySupervisedDataset
    )
    train_data = []
    with open(sys.argv[3], "r") as f:
        for line in f:
            train_data.append(line)
    dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=48)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=200)
    with open(os.path.join(sys.argv[4]), 'w') as f:
        with torch.no_grad():
            idx = 0
            for batch in data_loader:
                batch = {key: batch[key].cuda() for key in batch.keys()}
                embs = model(**batch)
                embs = embs.tolist()
                for emb in embs:
                    example = train_data[idx].strip()
                    f.write(example + '\001' + ','.join(list(map(str,emb))) + '\n')
                    idx += 1
