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
    _collate_level_fn, _flatten, _find_save_path,
    _load_state, _save_state,
)
from sklearn import metrics
import datetime

class TrainerLevel:

    TRAIN = 'train'
    FT = 'fine_tuning'
    SCORING = 'scoring'

    def __init__(self, args, tokenizer, model, dataset):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model      
        self.train_ds = dataset[0]
        self.eval_ds = dataset[1]
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"-- Running on {self.device}. --\n")
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.logging_step = args.logging_step
        self.warmup_ratio = args.warmup_ratio
        self.batch_size = args.batch_size
        self.path = p.join('./ckpt', f"{args.mode}")
        self.path = p.join(self.path, f"level_{args.level}")
        self.path = p.join(self.path, f"{args.feature_type}")
        if args.mode == "fine_tuning":
            self.path = p.join(self.path, f"{args.freeze}")
        os.makedirs(self.path, exist_ok=True)

        if self.args.scoring == 'no':
            args_file = open(p.join(self.path, 'args_and_logs.txt'), 'w')
            args_file.write(f"Start time : {datetime.datetime.now()}\n\n")

            args_dict = vars(args)
            dict_keys = args_dict.keys()
            dict_values = args_dict.values()

            args_file.write("Arguments :\n")
            for k, v in zip(dict_keys, dict_values):
                args_file.write(f"{k}:{v}\n")
            args_file.write("\n")
            args_file.close()

        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=5, collate_fn=_collate_level_fn)
        self.eval_loader = torch.utils.data.DataLoader(self.eval_ds, batch_size=self.batch_size, shuffle=False, num_workers=5, collate_fn=_collate_level_fn)
        
    def run(self, mode):
        if mode == TrainerLevel.TRAIN:
            print("Start train")
            self._train()

        if mode == TrainerLevel.FT:
            print("Start fine-tuning")
            self._fine_tuning()

        if mode == TrainerLevel.SCORING:
            print("Scoring")
            self._scoring()
        

    def _train(self):
        loss_fn = nn.MSELoss()
        best_loss = float('inf')
        
        print(f"-train feature {self.args.feature_type}-") 
        args_file = open(p.join(self.path, 'args_and_logs.txt'), 'a')
        args_file.write(f"Train_log :\n")
        args_file.close()

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
            for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(self.train_loader)):
                self.input_ids, self.attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                self.token_type_ids, self.labels = token_type_ids.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.model(input_ids=self.input_ids, attention_mask=self.attention_mask, token_type_ids=self.token_type_ids)

                loss = loss_fn(outputs.squeeze(), self.labels)
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.item()

                if step % self.logging_step == 0:                   
                    print(f"Train : epoch {e+1} batch_id {step+1} loss {(epoch_loss/(step+1)):.3f}")
            
            self.model.eval()
            for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(self.eval_loader)):
                self.input_ids, self.attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                self.token_type_ids, self.labels = token_type_ids.to(self.device), labels.to(self.device)

                outputs = self.model(input_ids=self.input_ids, attention_mask=self.attention_mask, token_type_ids=self.token_type_ids)
                    
                loss = loss_fn(outputs.squeeze(), self.labels)

                eval_loss += loss.item()
            eval_loss = eval_loss/len(self.eval_loader)
            print(f"Eval : epoch {e+1} loss {eval_loss:.3f}")

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_epoch = e+1
                os.makedirs(self.path, exist_ok=True)
                _save_state(self.model, self.optimizer, best_epoch, best_loss, self.path)
                args_file = open(p.join(self.path, 'args_and_logs.txt'), 'a')
                args_file.write(f"Epoch: {e+1}, Success to save model state. Best_epoch is {best_epoch}, Best_loss is {best_loss}\n")
                args_file.close()
                print(f"Success to save model state. Best_epoch is {best_epoch}, Best_loss is {best_loss}")
            else:
                args_file = open(p.join(self.path, 'args_and_logs.txt'), 'a')
                args_file.write(f"Epoch: {e+1}, Nothing improved. This epoch loss is {eval_loss}, Best_epoch is {best_epoch}, Best_loss is {best_loss}\n")
                args_file.close()
                print(f"Nothing improved. Best_epoch is {best_epoch}, Best_loss is {best_loss}")

        args_file = open(p.join(self.path, 'args_and_logs.txt'), 'a')
        args_file.write(f"\nFinish time : {datetime.datetime.now()}\n\n")
        args_file.close()

    def _fine_tuning(self):
        loss_fn = nn.MSELoss()
        best_loss = float('inf')
        
        print(f"-fine-tuning feature {self.args.feature_type}-") 
        self.model = self.model.to(self.device)
        total_steps = len(self.train_loader) * self.num_epochs
        warmup_step = int(total_steps * self.warmup_ratio)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                num_warmup_steps = warmup_step,
                                                num_training_steps = total_steps)  

        path = p.join('./ckpt', "level")
        path = p.join(path, f"level_{self.args.level}")
        path = p.join(path, f"{self.args.feature_type}")

        best_model_path = p.join(path, _find_save_path(path, 'pt'))
        self.model, self.optimzier, epoch, _ = _load_state(self.model, self.optimizer, best_model_path)
        
        if self.args.freeze == "yes":
            for name, child in self.model.named_children():
                for param in child.parameters():
                    if name == 'bert': 
                        param.requires_grad = False

        for e in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            eval_loss = 0.0
            for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(self.train_loader)):
                self.input_ids, self.attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                self.token_type_ids, self.labels = token_type_ids.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(input_ids=self.input_ids, attention_mask=self.attention_mask, token_type_ids=self.token_type_ids)
                score = torch.add(torch.mul(outputs, 4), 1) #rescale to range [1, 5]

                loss = loss_fn(outputs.squeeze(), self.labels)
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.item()

                if step % self.logging_step == 0:                   
                    print(f"Train : epoch {e+1} batch_id {step+1} loss {(epoch_loss/(step+1)):.3f}")
        
          
            self.model.eval()
            for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(self.eval_loader)):
                self.input_ids, self.attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                self.token_type_ids, self.labels = token_type_ids.to(self.device), labels.to(self.device)

                outputs = self.model(input_ids=self.input_ids, attention_mask=self.attention_mask, token_type_ids=self.token_type_ids)
                score = torch.add(torch.mul(outputs, 4), 1) #rescale to range [1, 5]

                loss = loss_fn(outputs.squeeze(), self.labels)

                eval_loss += loss.item()
            eval_loss = eval_loss/len(self.eval_loader)
            print(f"Eval : epoch {e+1} loss {eval_loss:.3f}")

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_epoch = e+1
                os.makedirs(self.path, exist_ok=True)
                _save_state(self.model, self.optimizer, best_epoch, best_loss, self.path)
                args_file = open(p.join(self.path, 'args_and_logs.txt'), 'a')
                args_file.write(f"Epoch: {e+1}, Success to save model state. Best_epoch is {best_epoch}, Best_loss is {best_loss}\n")
                args_file.close()
                print(f"Success to save model state. Best_epoch is {best_epoch}, Best_loss is {best_loss}")
            else:
                args_file = open(p.join(self.path, 'args_and_logs.txt'), 'a')
                args_file.write(f"Epoch: {e+1}, Nothing improved. This epoch loss is {eval_loss}, Best_epoch is {best_epoch}, Best_loss is {best_loss}\n")
                args_file.close()
                print(f"Nothing improved. Best_epoch is {best_epoch}, Best_loss is {best_loss}")

    def _scoring(self):
        self.convai_loader = self.train_loader
        self.empathetic_loader = self.eval_loader

        print(f"-Scoring {self.args.feature_type}-") 
        self.model = self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        path = p.join('./ckpt', f"{self.args.mode}")
        path = p.join(path, f"level_{self.args.level}")
        path = p.join(path, f"{self.args.feature_type}")
        if self.args.mode == 'fine_tuning':
            path = p.join(path, f"{self.args.freeze}")

        best_model_path = p.join(path, _find_save_path(path, 'pt'))
        self.model, self.optimzier, epoch, _ = _load_state(self.model, self.optimizer, best_model_path)

        convai_label = []
        convai_model = []
        empathetic_label = []
        empathetic_model = []

        self.model.eval()
        for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(self.convai_loader)):
            self.input_ids, self.attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            self.token_type_ids, self.labels = token_type_ids.to(self.device), labels.to(self.device)

            outputs = self.model(input_ids=self.input_ids, attention_mask=self.attention_mask, token_type_ids=self.token_type_ids)
            score = torch.add(torch.mul(outputs, 4), 1) #rescale to range [1, 5]

            convai_label.append(self.labels.tolist())
            convai_model.append(score.squeeze().tolist())

        self.model.eval()
        for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(self.empathetic_loader)):
            self.input_ids, self.attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            self.token_type_ids, self.labels = token_type_ids.to(self.device), labels.to(self.device)

            outputs = self.model(input_ids=self.input_ids, attention_mask=self.attention_mask, token_type_ids=self.token_type_ids)
            score = torch.add(torch.mul(outputs, 4), 1) #rescale to range [1, 5]

            empathetic_label.append(self.labels.tolist())
            empathetic_model.append(score.squeeze().tolist())

        convai_label, convai_model = list(_flatten(convai_label)), list(_flatten(convai_model))
        empathetic_label, empathetic_model = list(_flatten(empathetic_label)), list(_flatten(empathetic_model))

        data = {
            "convai_label":convai_label,
            "convai_model":convai_model,
            "empathetic_label":empathetic_label,
            "empathetic_model":empathetic_model
        }
        score_df = pd.DataFrame(data=data)


        save_path = p.join('./scoring', f"{self.args.mode}/level_{self.args.level}")
        save_path = p.join(save_path, f"{self.args.feature_type}")
        if self.args.mode == 'fine_tuning':
            save_path = p.join(save_path, f"{self.args.freeze}")
        os.makedirs(save_path, exist_ok=True)

        score_df.to_csv(p.join(save_path, f'{self.args.feature_type}_score.csv'), sep='\t')
