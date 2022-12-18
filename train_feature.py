import util.option as opt
import util.main_utils as main_utils
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from evaluator.eval_utils import _eval


tokens = ['[EOU]']

if __name__ == '__main__':
    args = opt.parse_train_feature_opt()
    print(args)
    main_utils.set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, model_max_length=512)
    
    for v in tokens:
        tokenizer.add_tokens(v)     
    
    # model = main_utils.get_model(args, tokenizer)
    # train_ds = main_utils.get_dataset(args, tokenizer, data_type='train')
    # eval_ds = main_utils.get_dataset(args, tokenizer, data_type='valid')
    # test_ds = main_utils.get_dataset(args, tokenizer, data_type='test')
    # dataset = [train_ds, eval_ds, test_ds]
    # trainer = main_utils.get_trainer(args, tokenizer, model, dataset)

    # trainer.run("train")
    # trainer.run("test")
    # trainer.run("eval")

    _eval(args)