import os
import os.path as p
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn.functional as F

class Level_Dataset(Dataset):
    def __init__(self, args, tokenizer, data_type):
        self.tokenizer = tokenizer
        path = p.join(f'./data/processed/level/level_{args.level}', f'{args.feature_type}')
        if data_type == "train":
            self.df = pd.read_csv(p.join(path, f'{data_type}.csv'), sep='\t')
            self.test_df = pd.read_csv(p.join(path, 'test.csv'), sep='\t')
            self.df = pd.concat([self.df, self.test_df])
        else:
            self.df = pd.read_csv(p.join(path, f'{data_type}.csv'), sep='\t')

        if data_type == "train":
            self.context = self.df['context'].str.lower().tolist()
            self.response = self.df['response'].str.lower().tolist()
            labels = self.df['label'].tolist()
            
        else:
            self.context = self.df['context'].str.lower().tolist()
            self.response = self.df['response'].str.lower().tolist()
            labels = self.df['label'].tolist()
            
        inputs = self.tokenizer(self.context, self.response, padding=True, truncation=True)
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.token_type_ids = inputs['token_type_ids']

        self.labels = []

        if args.level == 2:
            self.labels = labels
            
        if args.level == 3:
            for l in labels:
                if l == 3:
                    self.labels.append(1)
                if l == 2:
                    self.labels.append(0.5)
                if l == 1:
                    self.labels.append(0)

        if args.level == 5:
            for l in labels:
                if l == 5:
                    self.labels.append(1)
                if l == 4:
                    self.labels.append(0.5)
                if l == 3:
                    self.labels.append(0.25)
                if l == 2:
                    self.labels.append(0.25)
                if l == 1:
                    self.labels.append(0.25)
                if l == 0:
                    self.labels.append(0)


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.labels[idx])