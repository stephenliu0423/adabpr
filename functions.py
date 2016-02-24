

import numpy as np
from collections import defaultdict


def load_history_array(file_name):
    with open(file_name, "r") as inf:
        arr = np.array([line.split('$$')[:2] for line in inf], dtype=np.int32)
    return arr


def load_history_dict(file_name):
    user_history = defaultdict(list)
    with open(file_name, 'r') as inf:
        for line in inf:
            u, i, t = line.split('$$')
            user_history[int(u)].append(int(i))
    return user_history


def data_process(train_file, test_file):
    train_data = load_history_array(train_file)
    Te = load_history_dict(test_file)
    user_dict, items = defaultdict(list), []
    for u, i in train_data:
        user_dict[u].append(i)
        items.append(i)
    items = set(items)
    Tr, Tr_neg = {}, {}
    for u in user_dict:
        pos_set = set(user_dict[u])
        neg_set = list(items-pos_set)
        Tr[u] = {'items': np.array(list(pos_set)), 'num': len(pos_set)}
        Tr_neg[u] = {'items': np.array(neg_set), 'num': len(neg_set)}
    return train_data, Tr, Tr_neg, Te
