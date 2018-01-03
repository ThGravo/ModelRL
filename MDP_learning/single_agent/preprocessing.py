import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler
from fancyimpute import MICE, KNN
from copy import deepcopy


def standardise_memory(memory, state_size, action_size):
    states = np.vstack((memory[:, :state_size], memory[:, - state_size - 1:-1]))
    actions = memory[:, state_size: state_size + action_size]
    scaler = MinMaxScaler()
    states_scaled = scaler.fit_transform(states)
    actions_scaled = scaler.fit_transform(actions)
    memory[:, :state_size] = states_scaled[:len(memory), :state_size]
    memory[:, - state_size - 1:-1] = states_scaled[len(memory):, :state_size]
    memory[:, state_size: state_size + action_size] = actions_scaled


def make_mem_partial_obs(memory, state_size, partial_obs_rate):
    masks_states = np.random.choice([np.nan, 1.0], size=(len(memory), state_size),
                                    p=[partial_obs_rate, 1 - partial_obs_rate])
    masks_next_states = np.random.choice([np.nan, 1.0], size=(len(memory), state_size),
                                         p=[partial_obs_rate, 1 - partial_obs_rate])
    for i in range(len(memory)):
        if i == 0 or memory[i - 1, -1]:
            memory[i, :state_size] = masks_states[i] * memory[i, :state_size]
            memory[i, - state_size - 1:-1] = masks_next_states[i] * memory[i, - state_size - 1:-1]
        else:
            memory[i, :state_size] = memory[i - 1, -state_size - 1:-1]
            memory[i, - state_size - 1:-1] = masks_next_states[i] * memory[i, - state_size - 1:-1]

def impute_missing(memory, state_size, imputer):
    states = np.vstack((memory[:, :state_size], memory[:, - state_size - 1:-1]))
    states_imputed = imputer(n_imputations=50).complete(states)
    memory[:, :state_size] = states_imputed[:len(memory), :state_size]
    memory[:, - state_size - 1:-1] = states_imputed[len(memory):, :state_size]

def setup_batch_for_RNN(batch, sequence_length, state_size, action_size):
    batch_size = batch.shape[0]
    array_size = batch_size - sequence_length + 1
    x_seq = np.empty((array_size, sequence_length, state_size + action_size))
    y_seq = np.empty((array_size, state_size))
    actual_size = 0
    for jj in range(array_size):
        if not batch[jj:jj + sequence_length - 1, -1].any():
            x_seq[actual_size, ...] = batch[np.newaxis,
                                      jj:jj + sequence_length,
                                      :state_size + action_size]
            y_seq[actual_size, ...] = batch[np.newaxis,
                                      jj + sequence_length - 1,
                                      -state_size - 1:-1]
            actual_size += 1
    x_seq = np.resize(x_seq, (actual_size, sequence_length, state_size + action_size))
    y_seq = np.resize(y_seq, (actual_size, state_size))
    return x_seq, y_seq

