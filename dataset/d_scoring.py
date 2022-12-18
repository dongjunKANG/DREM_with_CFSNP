import os
import os.path as p
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn.functional as F

class Scoring_Dataset(Dataset):
    def __init__(self, args, tokenizer, data_type):
        self.tokenizer = tokenizer
        path = './data/processed/eval'
        self.df = pd.read_csv(p.join(path, f'{data_type}.csv'), sep='\t')

        self.context = self.df['ctx'].str.lower().tolist()
        self.response = self.df['hyp'].str.lower().tolist()
        self.labels = self.df['score'].tolist()

        inputs = self.tokenizer(self.context, self.response, padding=True, truncation=True)
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.token_type_ids = inputs['token_type_ids']


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.labels[idx])