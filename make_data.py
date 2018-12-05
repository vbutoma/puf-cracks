import time
import numpy as np
from hashlib import sha256
import pickle as pkl
import os
import random


dataset_size = 1e6
train_split = 0.8
val_split = 0.1
test_split = 0.1


def get_data_2(data_dir='data', file_name='Base8.txt'):
    data = []
    for row in open(os.path.join(data_dir, file_name)):
        if len(row.split()) == 2:
            inp, target = row.split()
            h = sha256(inp.encode()).hexdigest()
            s = ''
            for i in range(16):
                part = h[i * 2:2 * (i + 1)]
                x = int(part, 16)
                ss = ''
                for j in range(8):
                    ss += str((x >> j) & 1)
                s += ss
            row = [int(_) for _ in s]

            cur = row[0]
            for i in range(1, len(row)):
                cur ^= row[i]
                row[i] = cur
            row.append(1)
            data.append([np.array([1 if x else -1 for x in row]), float(target)])
            # data.append([np.array(row), float(target)])

    random.shuffle(data)
    train_count = int(dataset_size * train_split)
    val_count = int(dataset_size * val_split)
    test_count = int(dataset_size * test_split)
    return data[:train_count], data[train_count:train_count + val_count], \
           data[train_count + val_count:train_count + val_count + test_count]


def load_data(n, get_data_func=get_data_2, method=2, dump=False):
    start_time = time.time()
    file_name = 'Base{}.txt'.format(n)
    train_data, val_data, test_data = get_data_func(data_dir='data', file_name=file_name)
    if dump:
        with open('dumped3/train_data_{}_{}.pkl'.format(method, n), 'wb') as f:
            pkl.dump(train_data, f)
        with open('dumped3/val_data_{}_{}.pkl'.format(method, n), 'wb') as f:
            pkl.dump(val_data, f)
        with open('dumped3/test_data_{}_{}.pkl'.format(method, n), 'wb') as f:
            pkl.dump(test_data, f)

    print("Data loaded in {}".format(time.time() - start_time))
    return train_data, val_data, test_data


def test_sha():
    s = "10101010"
    h = sha256(s.encode()).hexdigest()
    print('Original:', h)
    bits = [_ for _ in bin(int(h, 16))[2:][:128]]
    print(''.join(bits))
    bits.reverse()
    print(''.join(bits))
    s = ''
    for i in range(16):
        part = h[i*2:2*(i + 1)]
        x = int(part, 16)
        ss = ''
        for j in range(8):
            ss += str((x >> j) & 1)
        # print(part, x, ss)
        s += ss
    print(s)


if __name__ == "__main__":
    train_data, val_data, test_data = load_data(n=128, get_data_func=get_data_2, method=3, dump=True)
    x_train, y_train = list(map(np.array, zip(*train_data)))
    x_val, y_val = list(map(np.array, zip(*test_data)))
    x_test, y_test = list(map(np.array, zip(*test_data)))
    x_val = x_val[:2000]
    y_val = y_val[:2000]

    # test_sha()
