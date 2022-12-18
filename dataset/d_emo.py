import os
import os.path as p
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn.functional as F

class FeatureEmo(Dataset):
    def __init__(self, args, tokenizer, data_type):
        self.tokenizer = tokenizer
        path = p.join('./data/processed/feature', f'{args.feature_type}')
        self.df = pd.read_csv(p.join(path, f'{data_type}_emo.csv'), sep='\t')

        self.utterance = self.df['utterance'].str.lower().tolist()
        inputs = self.tokenizer(self.utterance, padding=True)
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.emotion = self.df['emotion'].tolist()
        # self.emotion = [e-1 for e in self.emotion]


        one_hot_value = F.one_hot(torch.tensor(self.emotion), num_classes=7)
        self.labels = one_hot_value.tolist()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx], self.labels[idx])