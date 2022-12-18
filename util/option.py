import argparse

def parse_train_feature_opt():
    parser = argparse.ArgumentParser()

    #general settings
    parser.add_argument(
        '--seed',
        type=int,
        help='The seed for reproducibility (optional).')
    parser.add_argument(
        '--pretrained_model_name',
        choices=['bert-base-uncased'],
        help='Pretrained model name.')
    
    #trainer settings
    parser.add_argument(
        '--gpu',
        type=int,
        help='Number of gpu.')
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of epochs.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='The initial learning rate for training.')
    parser.add_argument(
        '--logging_step',
        type=int,
        help='The logging step for training.')
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        help='Warmup ratio for training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size for training.')

    parser.add_argument(
        '--feature_type',
        choices=['act', 'emo', 'topic'],
        help='Using feature name.')
    parser.add_argument(
        '--mode',
        choices=['feature', 'scoring'],
        help='Runing mode (currently supports `feature` / `scoring`).')

    args = parser.parse_args()
    return args

def parse_train_score_opt():
    parser = argparse.ArgumentParser()

    #general settings
    parser.add_argument(
        '--seed',
        type=int,
        help='The seed for reproducibility (optional).')
    parser.add_argument(
        '--pretrained_model_name',
        choices=['bert-base-uncased'],
        help='Pretrained model name.')
    
    #trainer settings
    parser.add_argument(
        '--gpu',
        type=int,
        help='Number of gpu.')
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of epochs.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='The initial learning rate for training.')
    parser.add_argument(
        '--logging_step',
        type=int,
        help='The logging step for training.')
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        help='Warmup ratio for training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size for training.')

    parser.add_argument(
        '--feature_type',
        choices=['act', 'emo', 'topic', 'all'],
        help='Using feature name.')
    parser.add_argument(
        '--mode',
        choices=['fine_tuning', 'level', 'scoring'],
        help='Runing mode (currently supports `fine-tuning` / `level`).')
    parser.add_argument(
        '--scoring',
        choices=['yes', 'no'],
        help='Runing mode (currently supports `yes` / `no`).')
    parser.add_argument(
        '--level',
        type=int,
        help='level for training.')
    parser.add_argument(
        '--freeze',
        choices=['yes', 'no'],
        help='Runing mode (currently supports `fine-tuning` / `level`).')

    args = parser.parse_args()
    return args

def parse_eval_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--feature_type',
        choices=['act', 'emo', 'topic'],
        help='Using feature name.')
    parser.add_argument(
        '--mode',
        choices=['fine_tuning', 'level'],
        help='Runing mode (currently supports `fine-tuning` / `level`).')
    parser.add_argument(
        '--freeze',
        choices=['yes', 'no'],
        help='Runing mode (currently supports `fine-tuning` / `level`).')
    parser.add_argument(
        '--level',
        type=int,
        help='level for training.')

    args = parser.parse_args()
    return args
