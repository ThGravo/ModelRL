from keras.layers import Dense, LSTM, GRU
from keras.optimizers import Adam
from keras.models import Sequential
import keras as K
import numpy as np


def build_regression_model(input_dim,
                           output_dim,
                           dim_multipliers=(6, 4),
                           activations=('relu', 'sigmoid'),
                           lr=.001):
    model = Sequential()
    model.add(Dense(1000, input_dim=input_dim, activation=activations[0]))
    for i in range(len(dim_multipliers) - 1):
        model.add(Dense(1000,
                        activation=activations[min(i + 1, len(activations) - 1)]))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['mse', 'mae', 'acc', 'mape'])
    model.summary()
    return model

def build_recurrent_regression_model(input_dim,
                                     output_dim,
                                     dim_multipliers=(6, 4),
                                     activations=('sigmoid', 'sigmoid'),
                                     lr=.001):
    num_hlayers = len(dim_multipliers)
    model = Sequential()
    model.add(LSTM(200,
                   input_shape=(None, input_dim),
                   activation=activations[0],
                   return_sequences=num_hlayers is not 1
                   ))
    for i in range(num_hlayers - 1):
        model.add(LSTM(500,
                       activation=activations[min(i + 1, len(activations) - 1)],
                       return_sequences=i is not num_hlayers - 2))
        # stacked LSTMs need to return a sequence
    model.add(Dense(output_dim, activation='relu'))
    model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['mse', 'mae', 'acc', 'mape'])
    model.summary()
    return model

def build_dmodel(input_dim,
                 dim_multipliers=(4, 4),
                 activations=('relu', 'relu'),
                 lr=.001):
    model = Sequential()
    model.add(Dense(500, input_dim=input_dim, activation=activations[0]))
    for i in range(len(dim_multipliers) - 1):
        model.add(Dense(500, activation=activations[i + 1]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    # model.summary()
    return model