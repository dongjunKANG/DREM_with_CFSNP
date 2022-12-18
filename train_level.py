import util.option as opt
import util.main_utils as main_utils
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from evaluator.eval_utils import _eval


tokens = ['[EOU]']

if __name__ == '__main__':
    args = opt.parse_train_score_opt()
    print(args)

    main_utils.set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    
    for v in tokens:
        tokenizer.add_tokens(v)     
    
    model = main_utils.get_model(args, tokenizer)
    if args.scoring == 'yes':
        convai_ds = main_utils.get_score_dataset(args, tokenizer, data_type='convai2')
        emphathetic_ds = main_utils.get_score_dataset(args, tokenizer, data_type='empatheticdialogues')
        dataset = [convai_ds, emphathetic_ds]

    if args.scoring == 'no':
        if args.mode == 'level':
            train_ds = main_utils.get_level_dataset(args, tokenizer, data_type='train')
            eval_ds = main_utils.get_level_dataset(args, tokenizer, data_type='valid')
            test_ds = main_utils.get_level_dataset(args, tokenizer, data_type='test')
            dataset = [train_ds, eval_ds, test_ds]    
        if args.mode == 'fine_tuning':
            train_ds = main_utils.get_level_dataset(args, tokenizer, data_type='train')
            eval_ds = main_utils.get_level_dataset(args, tokenizer, data_type='valid')
            dataset = [train_ds, eval_ds]

    trainer = main_utils.get_trainer(args, tokenizer, model, dataset)

    if args.scoring == 'no':
        if args.mode == 'level':
            trainer.run("train")
        if args.mode == 'fine_tuning':
            trainer.run("fine_tuning")
    if args.scoring == 'yes':
        trainer.run("scoring")