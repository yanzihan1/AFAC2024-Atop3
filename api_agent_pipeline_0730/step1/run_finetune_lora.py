# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import logging
import os
import pathlib
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, BitsAndBytesConfig, deepspeed
from utils import InfoNCETrainer, Qwen2ForANN
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_query_len: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_label_len: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                chat_template=TEMPLATE,
                tokenize=True,
                add_generation_prompt=False,
                padding=True,
                max_length=max_len,
                truncation=True,
            )
        )
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )

def preprocess_myself(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_query_len: int,
    max_label_len: int
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    input_ids_a = []
    attention_mask_a = []
    input_ids_b = []
    attention_mask_b = []
    input_ids_c = []
    attention_mask_c = []

    for i, msg in enumerate(messages):
        #print(msg['text'])
        #print(msg['label'])
        texts = msg.strip().split('\t')
        m_a = [
            {"role": "user", "content": texts[0]}
        ]
        text_a = tokenizer.apply_chat_template(
            m_a,
            tokenize=False,
            add_generation_prompt=True
        )
        m_b = [
            {"role": "user", "content": texts[1]}
        ]
        text_b = tokenizer.apply_chat_template(
            m_b,
            tokenize=False,
            add_generation_prompt=True
        )
        text_a = tokenizer(
                text_a,
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=max_query_len
            )
        text_b = tokenizer(
                text_b,
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=max_label_len
            )
        if len(texts) < 3:
            text_c = tokenizer(
                "",
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=max_label_len
            )
        else:
            m_c = [
                {"role": "user", "content": texts[2]}
            ]
            text_c = tokenizer.apply_chat_template(
                m_c,
                tokenize=False,
                add_generation_prompt=True
            )
            text_c = tokenizer(
                    text_c,
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                    max_length=max_label_len
                )
        input_ids_a.append([tokenizer.pad_token_id]*(max_query_len - len(text_a['input_ids'])) + text_a['input_ids'])
        input_ids_b.append([tokenizer.pad_token_id]*(max_label_len - len(text_b['input_ids'])) + text_b['input_ids'])
        input_ids_c.append([tokenizer.pad_token_id]*(max_label_len - len(text_c['input_ids'])) + text_c['input_ids'])
        attention_mask_a.append([0]* (max_query_len - len(text_a['input_ids'])) + text_a['attention_mask'])
        attention_mask_b.append([0]* (max_label_len - len(text_b['input_ids'])) + text_b['attention_mask'])
        attention_mask_c.append([0]* (max_label_len - len(text_c['input_ids'])) + text_c['attention_mask'])
    #print(texts)
    input_ids_a = torch.tensor(input_ids_a, dtype=torch.int)
    input_ids_b = torch.tensor(input_ids_b, dtype=torch.int)
    input_ids_c = torch.tensor(input_ids_c, dtype=torch.int)
    attention_mask_a  = torch.tensor(attention_mask_a, dtype=torch.int) 
    attention_mask_b  = torch.tensor(attention_mask_b, dtype=torch.int) 
    attention_mask_c  = torch.tensor(attention_mask_c, dtype=torch.int)                     

    return dict(
        input_ids_a=input_ids_a, 
        input_ids_b=input_ids_b, 
        input_ids_c=input_ids_c, 
        attention_mask_a=attention_mask_a,
        attention_mask_b=attention_mask_b,
        attention_mask_c=attention_mask_c,
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_query_len: int, max_label_len: int
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        messages = [example for example in raw_data]
        data_dict = preprocess_myself(messages, tokenizer, max_query_len, max_label_len)

        self.input_ids_a = data_dict["input_ids_a"]
        self.input_ids_b = data_dict["input_ids_b"]
        self.input_ids_c = data_dict["input_ids_c"]
        self.attention_mask_a = data_dict["attention_mask_a"]
        self.attention_mask_b = data_dict["attention_mask_b"]
        self.attention_mask_c = data_dict["attention_mask_c"]

    def __len__(self):
        return len(self.input_ids_a)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids_a=self.input_ids_a[i],
            attention_mask_a=self.attention_mask_a[i],
            input_ids_b=self.input_ids_b[i],
            attention_mask_b=self.attention_mask_b[i],
            input_ids_c=self.input_ids_c[i],
            attention_mask_c=self.attention_mask_c[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_query_len: int, max_label_len: int
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_label_len = max_label_len
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess_myself([self.raw_data[i]], self.tokenizer, self.max_query_len, self.max_label_len)
        ret = dict(
            input_ids_a=ret["input_ids_a"][0],
            attention_mask_a=ret["attention_mask_a"][0],
            input_ids_b=ret["input_ids_b"][0],
            attention_mask_b=ret["attention_mask_b"][0],
            input_ids_c=ret["input_ids_c"][0],
            attention_mask_c=ret["attention_mask_c"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_query_len,
    max_label_len
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_data = []
    with open(data_args.data_path, "r") as f:
        for line in f:
            train_data.append(line)
    train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_query_len=max_query_len, max_label_len=max_label_len)

    if data_args.eval_data_path:
        eval_data = []
        with open(data_args.eval_data_path, "r") as f:
            for line in f:
                eval_data.append(json.loads(line))
        eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if (
        getattr(training_args, "deepspeed", None)
        and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")

    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Load model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False

    model = Qwen2ForANN(model_args.model_name_or_path,config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=256,
        padding_side="left",
        use_fast=False,
    )

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_query_len=training_args.max_query_len, max_label_len=training_args.max_label_len
    )

    # Start trainer
    trainer = InfoNCETrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # `not training_args.use_lora` is a temporary workaround for the issue that there are problems with
    # loading the checkpoint when using LoRA with DeepSpeed.
    # Check this issue https://github.com/huggingface/peft/issues/746 for more information.
    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.use_lora
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    #torch.save(model.state_dict(), os.path.join(training_args.output_dir, 'pytorch_model.bin'))
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
    )


if __name__ == "__main__":
    train()
