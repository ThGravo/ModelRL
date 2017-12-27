from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.models import Sequential


# approximate Transition function
# state and action is input and successor state is output
def build_regression_model(input_dim,
                           output_dim,
                           state_size,
                           dim_multipliers=(16, 8),
                           activations=('sigmoid', 'sigmoid'),
                           lr=.001,
                           recurrent=False):
    num_hlayers = len(dim_multipliers)
    model = Sequential()
    if recurrent:
        model.add(LSTM(state_size * dim_multipliers[0],
                       input_shape=(None, input_dim),
                       activation=activations[0],
                       return_sequences=num_hlayers is not 1))
        for i in range(num_hlayers - 1):
            # stacked LSTMs need to return a sequence
            model.add(LSTM(state_size * dim_multipliers[i + 1],
                           activation=activations[min(i + 1, len(activations) - 1)],
                           return_sequences=i is not num_hlayers - 2))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['accuracy'])
    else:
        model.add(Dense(state_size * dim_multipliers[0], input_dim=input_dim, activation=activations[0]))
        for i in range(num_hlayers - 1):
            model.add(Dense(state_size * dim_multipliers[i + 1],
                            activation=activations[min(i + 1, len(activations) - 1)]))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary()
    return model


# approximate Done value
# state is input and reward is output
def build_dmodel(input_dim,
                 state_size,
                 dim_multipliers=(14, 8),
                 activations=('sigmoid', 'sigmoid'),
                 lr=.001):
    model = Sequential()
    model.add(Dense(state_size * dim_multipliers[0], input_dim=input_dim, activation=activations[0]))
    for i in range(len(dim_multipliers) - 1):
        model.add(Dense(state_size * dim_multipliers[i + 1], activation=activations[i + 1]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    # model.summary()
    return model


