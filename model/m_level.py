import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

class ModelLevel(nn.Module):
    def __init__(self, args, tokenizer, hidden_size=768):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = AutoModel.from_pretrained(args.pretrained_model_name)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.score_layer = nn.Linear(hidden_size, 1)

        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        score = self.score_layer(outputs.pooler_output)
        score = torch.sigmoid(score)
        return score
