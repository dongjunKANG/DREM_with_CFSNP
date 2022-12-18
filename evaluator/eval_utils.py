from sklearn import metrics
import pandas as pd
import os
import os.path as p

def _find_save_path(path, option):
    file_list = os.listdir(path)
    file_list_pt = [file for file in file_list if file.endswith(f".{option}")]
    sorted_file_list = sorted(file_list_pt)
    return sorted_file_list[-1]

def _eval(args):
    path = p.join('./ckpt', f"{args.mode}")
    path = p.join(path, f"{args.feature_type}")
    test_path = p.join(path, _find_save_path(path, 'csv'))
    results_df = pd.read_csv(test_path, sep='\t')
    print("-Evaluation performance-")
    label = results_df['label'].tolist()
    model = results_df['model'].tolist()
    print(metrics.classification_report(label, model, digits=2))