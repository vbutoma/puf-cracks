import pandas as pd
from functools import reduce
from hashlib import sha256
import numpy as np
import os
import random


def read_file_txt(file_path, N=8):

    data = dict()
    data['y'] = []
    for i in range(N):
        data['x{}'.format(i)] = []
    # todo: parse only unique
    with open(file_path, 'r') as file:
        for line in file.readlines():
            _ = line.split()
            x, target = _[0], int(_[1])
            for i in range(len(x)):
                data['x{}'.format(i)].append(int(x[i]))
            data['y'].append(target)
    df = pd.DataFrame.from_dict(data)
    return df


dataset_size = 1e6
train_split = 0.8
val_split = 0.1
test_split = 0.1


def get_data(data_dir='data', file_name='Base8.txt', preprocess=True, use_hash=True):
    data = []
    for row in open(os.path.join(data_dir, file_name)):
        if len(row.split()) == 2:
            inp, target = row.split()
            if use_hash:
                int_val = int(inp, 2)
                bytes_val = int_val.to_bytes(16, 'little', signed=False)
                inp = bin(int(sha256(bytes_val).hexdigest(), 16))[2:]
                inp = ''.join(['0']*(256 - len(inp))) + inp
            data.append([np.array([float(-1 if x == '0' else 1) for x in inp]), float(target)])
    if preprocess:
        new_data = []
        for x, y in data:
            s = np.zeros_like(x)
            lamb = reduce(lambda a, b: a ^ b, x, 0)
            for i in range(len(x)):
                s[i] = float(-1 if lamb == 1 else 1)
                lamb ^= x[i]
            new_data.append([s + [1], y])
        data = new_data

    random.shuffle(data)
    train_count = int(dataset_size*train_split)
    val_count = int(dataset_size*val_split)
    test_count = int(dataset_size*test_split)
    return data[:train_count], data[train_count:train_count+val_count], \
           data[train_count+val_count:train_count+val_count+test_count]


def get_data_2(data_dir='data', file_name='Base8.txt', preprocess=True, use_hash=True):
    data = []
    for row in open(os.path.join(data_dir, file_name)):
        if len(row.split()) == 2:
            inp, target = row.split()
            row = []
            n = len(inp)
            cur = int(inp[0])
            row.append(cur)
            for i in range(1, n):
                cur ^= int(inp[i])
                row.append(cur)
            data.append([np.array([1 if x else -1 for x in row]), float(target)])
            # cur = inp[n - 1]
            # for i in range(n-2, 0, -1):
            #     cur ^= inp[i]
            #     row.append(cur)
            # data.append([np.array([1 if x else -1 for x in row]), float(target)])
    # if preprocess:
    #     new_data = []
    #     for x, y in data:
    #         s = np.zeros_like(x)
    #         lamb = reduce(lambda a, b: a ^ b, x, 0)
    #         for i in range(len(x)):
    #             s[i] = float(-1 if lamb == 1 else 1)
    #             lamb ^= x[i]
    #         new_data.append([s, y])
    #     data = new_data

    random.shuffle(data)
    train_count = int(dataset_size * train_split)
    val_count = int(dataset_size * val_split)
    test_count = int(dataset_size * test_split)
    return data[:train_count], data[train_count:train_count + val_count], \
           data[train_count + val_count:train_count + val_count + test_count]

