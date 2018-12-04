from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout
from keras.regularizers import L1L2, l2
from keras.optimizers import Adam, SGD, RMSprop

from keras.metrics import binary_accuracy

import lightgbm as lgb

import math


def nn_model(input_dim, output_dim=1, lr=1e-1):
    optimizer = RMSprop(lr)
    model = Sequential()
    model.add(Dense(input_dim // 2, input_shape=(input_dim, ), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(input_dim // 4, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(input_dim // 8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=optimizer)
    return model


def build_logistic_model(input_dim, output_dim, lr=1e-2):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='sigmoid', kernel_regularizer=l2(0.)))
    # model.summary()
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='sgd')
    return model
