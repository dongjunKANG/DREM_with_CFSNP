import pandas as pd
import os
import os.path as p
import numpy as np
import scipy.stats

import util.option as opt
import util.main_utils as main_utils
from transformers import AutoTokenizer
import os

if __name__ == '__main__':
    args = opt.parse_eval_opt()
    print(args)


    save_path = p.join('./scoring', f"{args.mode}/level_{args.level}")
    save_path = p.join(save_path, f"{args.feature_type}")
    if args.mode == "fine_tuning":
        save_path = p.join(save_path, f"{args.freeze}")
    score_df = pd.read_csv(p.join(save_path, f'{args.feature_type}_score.csv'), sep='\t')

    convai_label, convai_model = score_df['convai_label'].tolist(), score_df['convai_model'].tolist()
    empathetic_label, empathetic_model = score_df['empathetic_label'].tolist(), score_df['empathetic_model'].tolist()

    for label, model, name in zip((convai_label, empathetic_label), (convai_model, empathetic_model), ("convai2", "empathetic")):
        # print(f"{name}_correlation")
        model_score = np.array(model)
        human_score = np.array(label)

        pearson = scipy.stats.pearsonr(model_score, human_score)
        spearman = scipy.stats.spearmanr(model_score, human_score)
        kendall = scipy.stats.kendalltau(model_score, human_score)

        print(f"{name}'s pearson correlation : {pearson[0]}, p-value : {pearson[1]}")
        print(f"{name}'s spearman correlation : {spearman[0]}, p-value : {spearman[1]}")
        print(f"{name}'s kendall correlation : {kendall[0]}, p-value : {kendall[1]}\n")