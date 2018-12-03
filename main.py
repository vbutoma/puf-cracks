__author__ = "Vitaly Butoma"

from utils import read_file_txt, get_data, get_data_2
import numpy as np
import pickle as pkl
import time

from nn import nn_model, build_logistic_model
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.utils import np_utils
import lightgbm as lgb
SIZE = 32
np.random.seed(42)


def load_data(load=False, n=SIZE, get_data_func=get_data_2, method=2, dump=False):
    start_time = time.time()
    if not load:
        file_name = 'Base{}.txt'.format(n)
        train_data, val_data, test_data = get_data_func(data_dir='data',
                                                        file_name=file_name, preprocess=True, use_hash=False)
        if dump:
            with open('dumped/train_data_{}_{}.pkl'.format(method, n), 'wb') as f:
                pkl.dump(train_data, f)
            with open('dumped/val_data_{}_{}.pkl'.format(method, n), 'wb') as f:
                pkl.dump(val_data, f)
            with open('dumped/test_data_{}_{}.pkl'.format(method, n), 'wb') as f:
                pkl.dump(test_data, f)
    else:
        with open("dumped/train_data_{}_{}.pkl".format(method, n), 'rb') as f:
            train_data = pkl.load(f)
        with open("dumped/val_data_{}_{}.pkl".format(method, n), 'rb') as f:
            val_data = pkl.load(f)
        with open("dumped/test_data_{}_{}.pkl".format(method, n), 'rb') as f:
            test_data = pkl.load(f)

    print("Data loaded in {}".format(time.time() - start_time))
    return train_data, val_data, test_data


def train_neural_net(n, x_train, y_train, x_val, y_val, x_test, y_test, epochs=100):
    model = nn_model(input_dim=x_train[0].shape[0])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=int(1e4), validation_data=(x_test, y_test))
    plt.plot(history.history['acc'])
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.savefig('nn_{}.png'.format(n))
    plt.show()
    score = model.evaluate(x_test, y_test, verbose=0)
    with open("logs3/nn_{}.txt".format(n), 'w') as file:
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        file.writelines('Test score:' + str(score[0]) + '\n')
        file.writelines('Test accuracy:' + str(score[1]) + '\n')


def train_logit_reg(n, x_train, y_train, x_val, y_val, x_test, y_test, epochs=10):
    number_of_classes = 2
    Y_train = np_utils.to_categorical(y_train, number_of_classes)
    Y_val = np_utils.to_categorical(y_val, number_of_classes)
    Y_test = np_utils.to_categorical(y_test, number_of_classes)
    # print(x_train[0].shape)
    model = build_logistic_model(input_dim=x_train[0].shape[0], output_dim=number_of_classes)
    # history = model.fit(x_train, Y_train, epochs=epochs, batch_size=int(1e4), validation_data=(x_val, Y_val))
    # model.save_weights('logit_weights.h5')
    # plt.plot(history.history['acc'])
    # plt.xlabel('epoch number')
    # plt.ylabel('accuracy')
    # plt.savefig('logit_{}.png'.format(n))
    # plt.show()
    model.load_weights('logit_weights.h5')
    score = model.evaluate(x_test, Y_test, verbose=0)
    res = model.predict_classes(x_test)
    good = 0
    for i in range(len(res)):
        if res[i] == y_test[i]:
            good += 1
    print(good, len(res))
    with open("logs/logit_{}.txt".format(n), 'w') as file:
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        file.writelines('Test score:' + str(score[0]) + '\n')
        file.writelines('Test accuracy:' + str(score[1]) + '\n')


def train_tree(n, x_train, y_train, x_val, y_val, x_test, y_test, nround=500):
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_val = lgb.Dataset(x_val, y_val)
    lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train)
    num_leaves = max(10, n // 2)
    bf = 5
    early_stop_freq = 2 * bf
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_error'],
        'num_leaves': num_leaves,
        'learning_rate': 0.05,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'verbose': 1
    }
    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=nround,
                    valid_sets=lgb_val,
                    early_stopping_rounds=early_stop_freq)

    with open('logs3/tree_{}.txt'.format(n), 'w') as f:
        f.writelines(str(gbm.best_score) + '\n')
        f.writelines(str(gbm.best_iteration) + '\n')
    gbm.save_model('models3/model_{}.txt'.format(n))


def run_tree(n=128):
    train_data, val_data, test_data = load_data(load=False, n=n, get_data_func=get_data_2, method=2, dump=True)
    x_train, y_train = list(map(np.array, zip(*train_data)))
    x_val, y_val = list(map(np.array, zip(*test_data)))
    x_test, y_test = list(map(np.array, zip(*test_data)))
    train_tree(n, x_train, y_train, x_val, y_val, x_test, y_test)


def run_logit(n=128):
    train_data, val_data, test_data = load_data(load=False, n=n, get_data_func=get_data_2, method=2, dump=True)
    x_train, y_train = list(map(np.array, zip(*train_data)))
    x_val, y_val = list(map(np.array, zip(*test_data)))
    x_test, y_test = list(map(np.array, zip(*test_data)))
    train_logit_reg(n, x_train, y_train, x_val, y_val, x_test, y_test, epochs=30)


def run_net(n=128):
    train_data, val_data, test_data = load_data(load=True, n=n, get_data_func=get_data_2, method=2, dump=False)
    x_train, y_train = list(map(np.array, zip(*train_data)))
    x_val, y_val = list(map(np.array, zip(*test_data)))
    x_test, y_test = list(map(np.array, zip(*test_data)))
    train_neural_net(n, x_train, y_train, x_val, y_val, x_test, y_test, epochs=30)


if __name__ == "__main__":
    sizes = [8 * (i + 1) for i in range(16)]
    print(sizes)
    sizes = [8]
    for _ in sizes:
        # run_tree(_)
        run_logit(_)
        # run_net(_)

