import os
import os.path as p
import torch
import torch.nn as nn

# Save & Load func.
def _save_state(model, optimizer, best_epoch, best_loss, path):
    save_path = p.join(path, f"epoch_{best_epoch}.pt")
    torch.save({
        "epoch" : best_epoch,
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "loss" : best_loss,
    }, save_path)

def _load_state(model, optimizer, best_model_path):
    try:
        checkpoint = torch.load(best_model_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print(f"Load checkpoint state. epoch is {epoch}, loss is {loss}")
    except:
        print("No checkpoint state exist.")
        epoch = 0
        loss = float('inf')
    return model, optimizer, epoch, loss

# Functions
def _collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.FloatTensor(labels)

def _collate_level_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    token_type_ids = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(token_type_ids), torch.FloatTensor(labels)

def _find_save_path(path, option):
    file_list = os.listdir(path)
    file_list_pt = [file for file in file_list if file.endswith(f".{option}")]
    sorted_file_list = sorted(file_list_pt)
    return sorted_file_list[-1]

def _flatten(lst):
    for i in lst:
        if isinstance(i, list):
            for v in _flatten(i):
                yield v
        else:
            yield i

