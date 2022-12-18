import os
import sys
import importlib
import random
import torch
import numpy as np

CLASS_NAME_MAP = {
    # Models
    'm_feature':'ModelFeature',
    'm_level':'ModelLevel',
    
    # Datasets
    'd_act':'FeatureAct',
    'd_emo':'FeatureEmo',
    'd_topic':'FeatureTopic',
    'd_level':'Level_Dataset',
    'd_fine_tuning':'Fine_tuning_Dataset',
    'd_scoring':'Scoring_Dataset',

    # Trainers
    't_feature':'TrainerFeature',
    't_level':'TrainerLevel',
}


def add_module_search_paths(search_paths: list) -> None:
    """Adds paths for searching python modules.
    """
    augmented_search_paths = search_paths
    for path in search_paths:
        for root, dirs, _ in os.walk(path):
            cur_dir_paths = [os.path.join(root, cur_dir) for cur_dir in dirs]
            augmented_search_paths.extend(cur_dir_paths)
    sys.path.extend(augmented_search_paths)

def get_class(module_name):
    class_name = CLASS_NAME_MAP[module_name]
    module = importlib.import_module(module_name)
    Class = getattr(module, class_name)
    return Class

def get_model(args, tokenizer):
    add_module_search_paths(['./model'])
    if args.mode =="fine_tuning" or args.mode == "scoring":
        Model = get_class(f'm_level')
    else:
        Model = get_class(f'm_{args.mode}')
    model = Model(args, tokenizer)
    return model

def get_dataset(args, tokenizer, data_type):
    add_module_search_paths(['./dataset'])
    Dataset = get_class(f'd_{args.feature_type}')
    dataset = Dataset(args, tokenizer, data_type)
    return dataset

def get_level_dataset(args, tokenizer, data_type):
    add_module_search_paths(['./dataset'])
    Dataset = get_class(f'd_{args.mode}')
    dataset = Dataset(args, tokenizer, data_type)
    return dataset

def get_score_dataset(args, tokenizer, data_type):
    add_module_search_paths(['./dataset'])
    Dataset = get_class(f'd_scoring')
    dataset = Dataset(args, tokenizer, data_type)
    return dataset

def get_trainer(args, tokenizer, model, dataset):
    add_module_search_paths(['./trainer'])
    if args.mode =="fine_tuning" or args.mode =="scoring":
        Trainer = get_class(f't_level')
    else:
        Trainer = get_class(f't_{args.mode}')
    trainer = Trainer(args, tokenizer, model, dataset)
    return trainer

def set_seed(seed):
    """Fixes randomness to enable reproducibility.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False