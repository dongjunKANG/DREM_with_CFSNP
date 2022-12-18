import os
import os.path as p
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from transformers import (
    get_linear_schedule_with_warmup
)
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from train_utils import (
    _collate_fn, _flatten, _find_save_path,
    _load_state, _save_state,
)
from sklearn import metrics


class TrainerFeature:

    TRAIN = 'train'
    TEST = 'test'
    EVAL = 'eval'

    def __init__(self, args, tokenizer, model, dataset):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model      
        self.train_ds = dataset[0]
        self.eval_ds = dataset[1]
        self.test_ds = dataset[2]
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"-- Running on {self.device}. --\n")
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.logging_step = args.logging_step
        self.warmup_ratio = args.warmup_ratio
        self.batch_size = args.batch_size
        self.path = p.join('./ckpt', f"{args.mode}")
        self.path = p.join(self.path, f"{args.feature_type}")

        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=5, collate_fn=_collate_fn)
        self.eval_loader = torch.utils.data.DataLoader(self.eval_ds, batch_size=self.batch_size, shuffle=False, num_workers=5, collate_fn=_collate_fn)
        self.test_loader = torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=5, collate_fn=_collate_fn)

        
    def run(self, mode):
        if mode == TrainerFeature.TRAIN:
            print("Start train")
            self._train()

        if mode == TrainerFeature.TEST:
            print("Start test")
            self._test()
        
        if mode == TrainerFeature.EVAL:
            print("Start evaluation")
            self._eval()

    def _train(self):
        loss_fn = nn.CrossEntropyLoss()
        best_loss = float('inf')
        
        print(f"-train feature {self.args.feature_type}-") 
        self.model = self.model.to(self.device)
        total_steps = len(self.train_loader) * self.num_epochs
        warmup_step = int(total_steps * self.warmup_ratio)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                num_warmup_steps = warmup_step,
                                                num_training_steps = total_steps)   

        for e in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            eval_loss = 0.0
            for step, (input_ids, attention_mask, labels) in enumerate(tqdm(self.train_loader)):
                self.input_ids, self.attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                self.labels = labels.to(self.device)
                self.optimizer.zero_grad()
                
                if self.args.feature_type == 'act':
                    outputs = self.model._act_forward(input_ids=self.input_ids, attention_mask=self.attention_mask)
                if self.args.feature_type == 'emo':
                    outputs = self.model._emo_forward(input_ids=self.input_ids, attention_mask=self.attention_mask)
                if self.args.feature_type == 'topic':
                    outputs = self.model._topic_forward(input_ids=self.input_ids, attention_mask=self.attention_mask)

                loss = loss_fn(outputs.squeeze(), self.labels)
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.item()

                if step % self.logging_step == 0:                   
                    print(f"Train : epoch {e+1} batch_id {step+1} loss {(epoch_loss/(step+1)):.3f}")
            
            self.model.eval()
            for step, (input_ids, attention_mask, labels) in enumerate(tqdm(self.eval_loader)):
                self.input_ids, self.attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                self.labels = labels.to(self.device)

                if self.args.feature_type == 'act':
                    outputs = self.model._act_forward(input_ids=self.input_ids, attention_mask=self.attention_mask)
                if self.args.feature_type == 'emo':
                    outputs = self.model._emo_forward(input_ids=self.input_ids, attention_mask=self.attention_mask)
                if self.args.feature_type == 'topic':
                    outputs = self.model._topic_forward(input_ids=self.input_ids, attention_mask=self.attention_mask)
                    
                loss = loss_fn(outputs.squeeze(), self.labels)

                eval_loss += loss.item()
            eval_loss = eval_loss/len(self.eval_loader)
            print(f"Eval : epoch {e+1} loss {eval_loss:.3f}")

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_epoch = e+1
                os.makedirs(self.path, exist_ok=True)
                _save_state(self.model, self.optimizer, best_epoch, best_loss, self.path)
                print(f"Success to save model state. Best_epoch is {best_epoch}, Best_loss is {best_loss}")
            else:
                print(f"Nothing improved. Best_epoch is {best_epoch}, Best_loss is {best_loss}")


    def _test(self):
        t_labels = []
        t_models = []

        self.model = self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        best_model_path = p.join(self.path, _find_save_path(self.path, 'pt'))
        self.model, self.optimzier, epoch, _ = _load_state(self.model, self.optimizer, best_model_path)
        
        self.model.eval()
        for step, (input_ids, attention_mask, labels) in enumerate(tqdm(self.test_loader)):
            self.input_ids, self.attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            self.labels = labels.to(self.device)

            if self.args.feature_type == 'act':
                outputs = self.model._act_forward(input_ids=self.input_ids, attention_mask=self.attention_mask)
            if self.args.feature_type == 'emo':
                outputs = self.model._emo_forward(input_ids=self.input_ids, attention_mask=self.attention_mask)
            if self.args.feature_type == 'topic':
                outputs = self.model._topic_forward(input_ids=self.input_ids, attention_mask=self.attention_mask)
            
            t_labels.append(self.labels.tolist())
            t_models.append(outputs.tolist())

        gold = []
        pred = []
        for la, mo in zip(t_labels, t_models):
            for l in la:
                gold.append(l.index(max(l)))
            for m in mo:
                pred.append(m.index(max(m)))
            
        t_labels, t_models = list(_flatten(t_labels)), list(_flatten(t_models))
        results = {
            'label':gold,
            'model':pred
        }
        results_df = pd.DataFrame(data=results)
        results_df.to_csv(p.join(self.path, f"epoch_{epoch}_results.csv"), sep='\t', na_rep="") 

    def _eval(self):
        test_path = p.join(self.path, _find_save_path(self.path, 'csv'))
        results_df = pd.read_csv(test_path, sep='\t')
        print("-Evaluation performance-")
        label = results_df['label'].tolist()
        model = results_df['model'].tolist()
        print(metrics.classification_report(label, model, digits=2))

          
