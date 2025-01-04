from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, Literal, List, Union

import math
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
import transformers
from transformers import BatchEncoding, Trainer, Qwen2PreTrainedModel, AutoModel, BitsAndBytesConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class Qwen2ForANN(Qwen2PreTrainedModel):
    def __init__(self,path,config):  
        super().__init__(config)
        self.config = config
        self.qwen = AutoModel.from_pretrained(
            path, config=config, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.dropout = torch.nn.Dropout(0.1)
        self.mlp = torch.nn.Linear(config.hidden_size,768).cuda()
    
    def forward(self, input_ids, attention_mask, stage='train'):
        qwen_output = self.qwen(input_ids, attention_mask)[0][:,-1,:]
        dim_reduced_vec = self.mlp(qwen_output)
        if stage == 'train':
            dim_reduced_vec = self.dropout(dim_reduced_vec)
        normalized_vec = F.normalize(dim_reduced_vec, p=2, dim=1)
        return normalized_vec
    
    def enable_input_require_grads(self,**kwargs):
        self.qwen.enable_input_require_grads(**kwargs)

    def gradient_checkpointing_enable(self,**kwargs):
        self.qwen.gradient_checkpointing_enable(**kwargs)
        

class InfoNCETrainer(Trainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        **kwargs,
    ):
        
        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """
        self.data_collator = None
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }
        # prepare dataloader
        data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

        return data_loader
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        inputs_a = {key.replace("_a",""): batch[key] for key in ["input_ids_a", "attention_mask_a"]}
        inputs_b = {key.replace("_b",""): batch[key] for key in ["input_ids_b", "attention_mask_b"]}
        inputs_c = {key.replace("_c",""): batch[key] for key in ["input_ids_c", "attention_mask_c"]}
        emb_a = model(**inputs_a)
        emb_b = model(**inputs_b)
        emb_c = model(**inputs_c)

        embedding = torch.concat([emb_a,emb_b,emb_c],axis=0).to(device=emb_a.device)
        sim = torch.matmul(embedding,embedding.T).to(device=emb_a.device)
        sim = (sim - torch.eye(sim.shape[0]).to(device=emb_a.device) * 1e8) * 20
        sim = sim.index_select(0, torch.arange(0, emb_a.shape[0] * 2).to(device=emb_a.device))
        idxs_1 = torch.eye(emb_a.shape[0]).to(device=emb_a.device)
        idxs_0 = torch.zeros([emb_a.shape[0], emb_a.shape[0]]).to(device=emb_a.device)

        y_pred = torch.concat([torch.concat([idxs_0, idxs_1, idxs_0], axis=1), torch.concat([idxs_1, idxs_0, idxs_0], axis=1)], axis=0)

        loss_fn = CrossEntropyLoss()
        loss = loss_fn(sim,y_pred)
        metrics['loss'] = loss.detach().mean().cpu()
        return loss, metrics


    def compute_loss(self, model, inputs,return_outputs=False):
        """
            batch = [batch, training_stage, seq_len]
        """
        #compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        compute_loss_context_manager = torch.cuda.amp.autocast
        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        if return_outputs:
            return (loss, metrics)
        return loss
    




        
