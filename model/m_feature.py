import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

class ModelFeature(nn.Module):
    def __init__(self, args, tokenizer, hidden_size=768):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = AutoModel.from_pretrained(args.pretrained_model_name)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.act_layer = nn.Linear(hidden_size, 4)
        self.emo_layer = nn.Linear(hidden_size, 7)
        self.topic_layer = nn.Linear(hidden_size, 10)
        
    def _act_forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        score = self.act_layer(outputs.pooler_output)
        score = F.softmax(score, dim=1)
        return score

    def _emo_forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        score = self.emo_layer(outputs.pooler_output)
        score = F.softmax(score, dim=1)
        return score

    def _topic_forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        score = self.topic_layer(outputs.pooler_output)
        score = F.softmax(score, dim=1)
        return score