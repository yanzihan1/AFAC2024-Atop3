import faiss
import argparse
import numpy as np
from dataclasses import dataclass, field
from torch.utils.data import (DataLoader, Dataset)
from typing import Dict
from utils import Qwen2ForANN
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
import sys
import transformers
from peft import PeftModel, get_peft_model

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 对文档中的某一行进行处理，获取该样本的bidword及对应的embedding向量
def get_bid_emb_from_line(doc, dim):
    # 为确保bidword中出现逗号时能够正确处理
    bid,emb = doc.split('\001')
    emb = np.array(emb.split(','),dtype=np.float32)
    return bid, emb

# 获取embedding向量
def get_bids_emb_from_file(path, dim):
    res = []
    bids = []
    with open(path,'r') as f:
        # 遍历文件中的每一行，逐行获取向量
        for line in f:
            bid,emb = get_bid_emb_from_line(line.strip(), dim)
            res.append(emb)
            bids.append(bid)
            
    res = np.array(res).astype("float32")
    #res = res.reshape(-1,res.shape[-1])
    # 返回bidwords 及 embedding结果
    return bids, res

def get_bids_from_file(path):
    bids = []
    with open(path,'r') as f:
        # 遍历文件中的每一行，逐行获取向量
        for line in f:
            bid = line.strip()
            bids.append(bid)
    #res = res.reshape(-1,res.shape[-1])
    # 返回bidwords 及 embedding结果
    return bids

def ann_search(index, emb, k):
    D, I = index.search(emb, k)
    return D, I

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
       
    #print(texts)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The inference model path")
    parser.add_argument("--lora_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The lora adapter of inference model path")
    parser.add_argument("--input_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The output data path. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--dim",
                        default=768,
                        type=int,
                        required=False,
                        help="The dim of input vector")
    parser.add_argument("--top_index_path",
                        default=None,
                        type=str,
                        required=False,
                        help="The path of top index")
    parser.add_argument("--base_index_path",
                        default=None,
                        type=str,
                        required=False,
                        help="The path of base index")
    parser.add_argument("--top_dict",
                        default=False,
                        type=bool,
                        required=False,
                        help="bool value of top search")
    parser.add_argument("--base_dict",
                        default=False,
                        type=bool,
                        required=False,
                        help="bool value of base search")
    parser.add_argument("--top_corpus_path",
                        default="shrink_corpus.emb",
                        type=str,
                        required=False,
                        help="The path of top corpus")
    parser.add_argument("--base_corpus_path",
                        default="extend_corpus.emb",
                        type=str,
                        required=False,
                        help="The path of base corpus")
    parser.add_argument("--top_k",
                        default=10,
                        type=int,
                        required=False,
                        help="The number of bidwords per query in top search")
    parser.add_argument("--base_k",
                        default=90,
                        type=int,
                        required=False,
                        help="The number of bidwords per query in base search")
    parser.add_argument("--ann_threshold",
                        default=0.7,
                        type=float,
                        required=False,
                        help="The threshold score of ann search")
    parser.add_argument('--output_format_type',
                        type=str,
                        default='\t',
                        help="The format of output type")
    parser.add_argument('--prefix',
                        type=str,
                        default='',
                        help="To ensure the isolation of different corpora.")
    parser.add_argument('--seed',
                        type=int,
                        default=2024,
                        help="random seed for initialization")
    
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    model_path = args.model_path
    lora_path = args.lora_path
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
    all_queries = []
    with open(args.input_path, "r") as f:
        for line in f:
            all_queries.append(line)
    dataset = dataset_cls(all_queries, tokenizer=tokenizer, max_len=64)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=64)
    
    faiss.omp_set_num_threads(32)
    if args.top_dict:
        if args.top_index_path:
            top_keywords = get_bids_from_file(args.top_corpus_path)
            top_index = faiss.read_index(args.top_index_path)
        else:
            top_keywords, top_vectors = get_bids_emb_from_file(args.top_corpus_path,args.dim)
            top_index = faiss.index_factory(args.dim,'HNSW128',faiss.METRIC_INNER_PRODUCT)
            top_index.hnsw.efConstrunction=256
            top_index.hnsw.efSearch=200
            top_index.train(top_vectors)
            top_index.add(top_vectors)
            # 保存索引到文件
            faiss.write_index(top_index, f"{args.prefix}top_index.idx")
        print("Load top dict SUCCESS.",flush=True)
    if args.base_dict:
        if args.base_index_path:
            base_keywords = get_bids_from_file(args.base_corpus_path)
            base_index = faiss.read_index(args.base_index_path)
        else:
            base_keywords, base_vectors = get_bids_emb_from_file(args.base_corpus_path,args.dim)
            base_index = faiss.index_factory(args.dim,'HNSW128',faiss.METRIC_INNER_PRODUCT)
            base_index.hnsw.efConstrunction=256
            base_index.hnsw.efSearch=200
            base_index.train(base_vectors)
            base_index.add(base_vectors)
            # 保存索引到文件
            faiss.write_index(base_index, f"{args.prefix}base_index.idx")
        print("Load base dict SUCCESS.",flush=True)
    
    with open(args.output_path, 'w') as f:
        with torch.no_grad():
            idx = 0
            res_keywords = []
            for batch in data_loader:
                batch = {key: batch[key].cuda() for key in batch.keys()}
                embs = model(**batch, stage='eval')
                embs_np = np.array(embs.reshape(-1,args.dim).tolist()).astype("float32")
                if args.top_dict:
                    top_dis, top_keywords_idx = ann_search(top_index, embs_np, args.top_k)
                if args.base_dict:
                    base_dis, base_keywords_idx = ann_search(base_index, embs_np, args.base_k)
                if args.top_dict and args.base_dict:
                    keywords_idx = (top_keywords_idx, base_keywords_idx)
                    dis = (top_dis, base_dis)
                elif args.top_dict:
                    keywords_idx = top_keywords_idx
                    dis = top_dis
                elif args.base_dict:
                    keywords_idx = base_keywords_idx
                    dis = base_dis
                else:
                    print("Error, please set top_dict or base_dict to be True!")
                    exit(-1)
                batch_keywords = []
                if args.top_dict and args.base_dict:
                    for item_keywords, item_dis in zip(keywords_idx[0], dis[0]):
                        batch_keywords.append([(top_keywords[keyword_idx],item_dis) for keyword_idx, item_dis in zip(item_keywords,item_dis)])
                    for base_idx, (item_keywords, item_dis) in enumerate(zip(keywords_idx[1], dis[1])):
                        batch_keywords[base_idx].extend([(base_keywords[keyword_idx],item_dis) for keyword_idx, item_dis in zip(item_keywords,item_dis)])
                
                elif args.top_dict:
                    for item_keywords, item_dis in zip(keywords_idx, dis):
                        batch_keywords.append([(top_keywords[keyword_idx],item_dis) for keyword_idx, item_dis in zip(item_keywords,item_dis)])
                else:
                    for item_keywords, item_dis in zip(keywords_idx[1], dis[1]):
                        batch_keywords.append([(base_keywords[keyword_idx],item_dis) for keyword_idx, item_dis in zip(item_keywords,item_dis)])
                for inner_idx in range(len(batch_keywords)):
                    query_cache = set()
                    for keyword in batch_keywords[inner_idx]:
                        if keyword[1] > args.ann_threshold and keyword[0] not in query_cache:
                            query_cache.add(keyword[0])
                            if args.output_format_type == '\002':
                                continue
                            else:
                                f.write(all_queries[idx].strip()+'\t' + keyword[0] + '\t' + str(keyword[1]) + '\n')
                    if args.output_format_type == '\002':
                        f.write(all_queries[idx].strip() + '\t' + '\002'.join(list(query_cache)) + '\n')
                    idx += 1
                
