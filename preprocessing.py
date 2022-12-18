import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os
import os.path as p
import numpy as np
import pandas as pd
import itertools
from more_itertools import locate
import ast
import json
import random

def load_dailydialog(data_path):
    """
    Funct:
        - Load dataframe
        - Simple preprocessing
    """

    text_path = p.join(data_path, "dialogues_text.txt")
    emo_path = p.join(data_path, "dialogues_emotion.txt")
    topic_path = p.join(data_path, "dialogues_topic.txt")
    act_path = p.join(data_path, "dialogues_act.txt")

    with open(text_path) as f_text:
        text = f_text.read().splitlines()
    with open(emo_path) as f_emo:
        emotion = f_emo.read().splitlines()
    with open(topic_path) as f_topic:
        topic = f_topic.read().splitlines()
    with open(act_path) as f_act:
        act = f_act.read().splitlines()

    emotion = [e.split() for e in emotion]
    act = [a.split() for a in act]
    topic = [tp.split() for tp in topic]
    
    tp = []
    for i, emo in enumerate(emotion):
        tp.append(topic[i]*len(emo))

    return text, emotion, tp, act

def make_raw_dataframe(src_path):
    text, emotion, topic, act = load_dailydialog(src_path)

    text = [t.split("__eou__") for t in text]
    text = [t[:-1] for t in text]
    text = [[s.strip() for s in t] for t in text]
    context = [t[:-1] for t in text]
    context = [(" [EOU] ").join(t) for t in context]
    response = [t[-1:] for t in text]
    response = [("").join(t) for t in response]

    context_tp = [t[:-1] for t in topic]
    context_emo = [e[:-1] for e in emotion]
    context_act = [a[:-1] for a in act]
    res_tp = [t[-1:] for t in topic]
    res_emo = [e[-1:] for e in emotion]
    res_act = [a[-1:] for a in act]

    context_tp = [list(map(int, t)) for t in context_tp]
    context_emo = [list(map(int, t)) for t in context_emo]
    context_act = [list(map(int, t)) for t in context_act]
    res_tp = [list(map(int, t)) for t in res_tp]
    res_emo = [list(map(int, t)) for t in res_emo]
    res_act = [list(map(int, t)) for t in res_act]

    df = pd.DataFrame(list(zip(context, response, context_tp, context_emo, context_act, res_tp, res_emo, res_act)), 
            columns=['context', 'response', 'con_topic', 'con_emotion', 'con_act', 'res_topic', 'res_emotion', 'res_act'])
    
    df['label'] = 1

    train_raw_df = df.sample(frac=0.80, random_state=42)
    temp_df = df.drop(train_raw_df.index)
    valid_raw_df = temp_df.sample(frac=0.5, random_state=42)
    test_raw_df = temp_df.drop(valid_raw_df.index)

    raw_path = p.join(src_path, "raw/")

    for df, name in zip((train_raw_df, valid_raw_df, test_raw_df), ("train", "valid", "test")):
        df = df.reset_index(drop=True)
        df.to_csv(p.join(raw_path, f"{name}_raw.csv"), sep='\t', na_rep="")

def make_single_train_data():
    data_path = "./data/processed"
    uttr_df = pd.read_csv(p.join(data_path, f"source/single_uttr.csv"), sep='\t')

    tr_emo_df, va_emo_df = train_test_split(uttr_df, test_size=0.2, shuffle=True, stratify=uttr_df['emotion'])
    va_emo_df, te_emo_df = train_test_split(va_emo_df, test_size=0.5, shuffle=True, stratify=va_emo_df['emotion'])

    tr_act_df, va_act_df = train_test_split(uttr_df, test_size=0.2, shuffle=True, stratify=uttr_df['act'])
    va_act_df, te_act_df = train_test_split(va_act_df, test_size=0.5, shuffle=True, stratify=va_act_df['act'])

    tr_topic_df, va_topic_df = train_test_split(uttr_df, test_size=0.2, shuffle=True, stratify=uttr_df['topic'])
    va_topic_df, te_topic_df = train_test_split(va_topic_df, test_size=0.5, shuffle=True, stratify=va_topic_df['topic'])

    tr_emo_df.to_csv(p.join(data_path, f"raw/train_emo.csv"), sep='\t')
    va_emo_df.to_csv(p.join(data_path, f"raw/valid_emo.csv"), sep='\t')
    te_emo_df.to_csv(p.join(data_path, f"raw/test_emo.csv"), sep='\t')

    tr_act_df.to_csv(p.join(data_path, f"raw/train_act.csv"), sep='\t')
    va_act_df.to_csv(p.join(data_path, f"raw/valid_act.csv"), sep='\t')
    te_act_df.to_csv(p.join(data_path, f"raw/test_act.csv"), sep='\t')

    tr_topic_df.to_csv(p.join(data_path, f"raw/train_topic.csv"), sep='\t')
    va_topic_df.to_csv(p.join(data_path, f"raw/valid_topic.csv"), sep='\t')
    te_topic_df.to_csv(p.join(data_path, f"raw/test_topic.csv"), sep='\t')


def make_3_level_data(data_path, uttr_df):
    # data_path = "./data/processed"
    # uttr_df = pd.read_csv(p.join(data_path, f"source/single_uttr.csv"), sep='\t')
    train_df = pd.read_csv(p.join(data_path, "source/train.csv"), sep='\t')
    valid_df = pd.read_csv(p.join(data_path, "source/valid.csv"), sep='\t')
    test_df = pd.read_csv(p.join(data_path, "source/test.csv"), sep='\t')

    uttr = uttr_df['utterance'].tolist()
    uttr_top = uttr_df['topic'].tolist()
    uttr_emo = uttr_df['emotion'].tolist()
    uttr_act = uttr_df['act'].tolist()



    for df, name in zip((train_df, valid_df, test_df), ('train', 'valid', 'test')):
        print(f"Making {name} dataframe...")
        for feature in ('topic', 'emotion', 'act'):
            print(f"Making {feature}...")
            ctx = []
            res = []
            c_feature = []
            r_feature = []
            label = []

            context = df['context'].tolist()
            response = df['response'].tolist()
            con_topic = df['con_topic'].tolist()
            con_emotion = df['con_emotion'].tolist()
            con_act = df['con_act'].tolist()
            res_topic = df['res_topic'].tolist()
            res_emotion = df['res_emotion'].tolist()
            res_act = df['res_act'].tolist()

            if feature == 'topic':
                for i, r in enumerate(response):  
                    topic = ast.literal_eval(res_topic[i])[0]

                    same_df = uttr_df.groupby('topic').get_group(topic)
                    not_same_df = uttr_df.drop(same_df.index)
                    same_df = same_df.sample(n=1)
                    not_same_df = not_same_df.sample(n=1)
                    # indices = list(locate(uttr_top, lambda x: x == topic))
                    # idx = random.choice(indices)
                    # not_indices = list(locate(uttr_top, lambda x: x != topic))
                    # not_idx = random.choice(not_indices)
                    

                    ctx.append(context[i])
                    res.append(response[i])
                    c_feature.append(con_topic[i])
                    r_feature.append(res_topic[i])
                    label.append(3)

                    ctx.append(context[i])
                    # res.append(uttr[idx])
                    res.append(same_df['utterance'])
                    c_feature.append(con_topic[i])
                    r_feature.append(same_df['topic'])
                    label.append(2)

                    ctx.append(context[i])
                    # res.append(uttr[not_idx])
                    res.append(not_same_df['utterance'])
                    c_feature.append(con_topic[i])
                    # r_feature.append(uttr_top[not_idx])
                    r_feature.append(not_same_df['topic'])
                    label.append(1)

                    data = {
                        'context':ctx,
                        'response':res,
                        'con_topic':c_feature,
                        'res_topic':r_feature,
                        'label':label
                    }
                    level_df = pd.DataFrame(data=data)
                    level_df.to_csv(p.join(data_path, f"level/level_3/{feature}/{name}.csv"), sep='\t')
                    

def make_negative_sample(df, uttr_df):
    contexts, responses = df['context'].tolist(), df['response'].tolist()
    res_tps = df['res_topic'].tolist()
    res_emos = df['res_emotion'].tolist()
    res_acts = df['res_act'].tolist()                       # res_tp : ['[1]',...] res_emo : ['[4]'...]
    res_tps = [ast.literal_eval(t) for t in res_tps]            # res_tp : [[1],...] 
    res_emos = [ast.literal_eval(t) for t in res_emos]          # res_emo : [[4]...]
    res_acts = [ast.literal_eval(t) for t in res_acts]  
    res_tps = list(itertools.chain(*res_tps))                   # res_tp : [1, ...]
    res_emos = list(itertools.chain(*res_emos))                 # res_emo : [4, ...]
    res_acts = list(itertools.chain(*res_acts))                                                    
    
    uttr = uttr_df['utterance'].tolist()
    
    _c, _r, _label = [], [], []
    _tp, _emo, _act = [], [], []

    for context, response, res_tp, res_emo, res_act in zip(contexts, responses, res_tps, res_emos, res_acts):
        

        try:
            idx = uttr.index(response)
            filter_df = uttr_df.drop(idx)
        except:
            pass

        neg = filter_df[(filter_df['topic'] == res_tp) & (filter_df['emotion'] == res_emo) & (filter_df['act'] == res_act)]
        if len(neg['utterance'].tolist()) != 0:
            _c.append(context)
            _r.append(response)
            _label.append(3)
            _tp.append(res_tp)
            _emo.append(res_emo)
            _act.append(res_act)

            sample = neg.sample(n=1, replace=True)
            _c.append(context)
            _r.append(sample.iloc[0]['utterance'])
            _label.append(2)
            _tp.append(sample.iloc[0]['topic'])
            _emo.append(sample.iloc[0]['emotion'])
            _act.append(sample.iloc[0]['act'])

            sample = filter_df[(filter_df['topic'] != res_tp) & (filter_df['emotion'] != res_emo) & (filter_df['act'] != res_act)].sample(n=1, replace=True)
            _c.append(context)
            _r.append(sample.iloc[0]['utterance'])
            _label.append(1)
            _tp.append(sample.iloc[0]['topic'])
            _emo.append(sample.iloc[0]['emotion'])
            _act.append(sample.iloc[0]['act'])

            

    data = {
        "context":_c,
        "response":_r,
        "label":_label,
        "topic":_tp,
        "emotion":_emo,
        "act":_act
    }
    df = pd.DataFrame(data=data)

        # _batch = list(zip(_c, _r, _label))
        # random.shuffle(_batch)
        # _c, _r, _label = zip(*_batch)
        # c.append(_c)
        # r.append(_r)
        # label.append(_label)

    # c = list(itertools.chain(*c))
    # r = list(itertools.chain(*r))
    # label = list(itertools.chain(*label))
    # df = pd.DataFrame(list(zip(c, r, label)), 
    #         columns=['context', 'response', 'label'])

    return df


def main():
    data_path = "./data/processed"

    # tr_df = pd.read_csv(p.join(data_path, f"raw/train_raw.csv"), sep='\t')
    # va_df = pd.read_csv(p.join(data_path, f"raw/valid_raw.csv"), sep='\t')
    # te_df = pd.read_csv(p.join(data_path, f"raw/test_raw.csv"), sep='\t')

    # df = pd.concat([tr_df, va_df, te_df])
    # df = df.drop(columns='label')
    # df = df.reset_index(drop=True)
    # df = df.drop(columns='Unnamed: 0')
    # df.to_csv(p.join(data_path, "raw/total.csv"), sep='\t', na_rep="")

    # train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=True, stratify=df['res_topic'])
    # valid_df, test_df = train_test_split(valid_df, test_size=0.5, shuffle=True, stratify=valid_df['res_topic'])

    # train_df.to_csv(p.join(data_path, "raw/train.csv"), sep='\t', na_rep="")
    # valid_df.to_csv(p.join(data_path, "raw/valid.csv"), sep='\t', na_rep="")
    # test_df.to_csv(p.join(data_path, "raw/test.csv"), sep='\t', na_rep="")

    # tr_single_df = pd.read_csv(p.join(data_path, f"raw/train_single.csv"), sep='\t')
    # va_single_df = pd.read_csv(p.join(data_path, f"raw/valid_single.csv"), sep='\t')
    # te_single_df = pd.read_csv(p.join(data_path, f"raw/test_single.csv"), sep='\t')

    # single_df = pd.concat([tr_single_df, va_single_df, te_single_df])
    # single_df = single_df.drop(columns='Unnamed: 0')
    # single_df.to_csv(p.join(data_path, "raw/single_uttr.csv"), sep='\t', na_rep="")

    
    # ft_df = pd.read_csv(p.join(data_path, "eval/dailydialog.csv"), sep='\t')
    # ft_train_df = ft_df.sample(frac=0.9, random_state=42)
    # ft_valid_df = ft_df.drop(ft_train_df.index)
    # ft_train_df.to_csv(p.join(data_path, "eval/ft_train.csv"), sep='\t', na_rep="")
    # ft_valid_df.to_csv(p.join(data_path, "eval/ft_valid.csv"), sep='\t', na_rep="")

    train_df = pd.read_csv(p.join(data_path, "source/train.csv"), sep='\t')
    valid_df = pd.read_csv(p.join(data_path, "source/valid.csv"), sep='\t')
    test_df = pd.read_csv(p.join(data_path, "source/test.csv"), sep='\t')
    uttr_df = pd.read_csv(p.join(data_path, "source/single_uttr.csv"), sep='\t')

    save_path = p.join(data_path, f"level/level_3/all")
    os.makedirs(save_path, exist_ok=True)
    for df, name in zip((train_df, valid_df, test_df), ("train", "valid", "test")):
        print(f"generating {name} data...")
        gen_df = make_negative_sample(df, uttr_df)
        gen_df.to_csv(p.join(save_path, f"{name}.csv"), sep='\t')


if __name__ == '__main__':
    main()
