import tensorflow as tf
from collections import deque
from keras.layers import Dense, LSTM, GRU
from keras.optimizers import Adam
from keras.models import Sequential
import keras as K
from keras.callbacks import TensorBoard
import gym
import random
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('GTK3Cairo', warn=False, force=True)
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

'''
GRID SEARCH RESULT
Best parameter set was 
{'learning_rate': 0.001, 'tmodel_activations': ('relu', 'sigmoid'), 'tmodel_dim_multipliers': (6, 6)}


def fillWithMean(df):
    if all(df.isnull()): return df.fillna(0.0)
    return df.fillna(df.mean())

def fillWithEWMean(df):
    if all(df.isnull()): return df.fillna(0.0)
    return df.fillna(df.ewm(alpha=0.9, ignore_na=True, adjust=False).mean())

def forwardFill(df):
    if all(df.isnull()): return df.fillna(0.0)
    return df.ffill()




x = fillWithEWMean(a)

print(fillWithEWMean(a))

print(forwardFill(a))

print(x.as_matrix()[-1])

print(x.values.tolist()[-1])

print(x[-1])


import gym
import numpy as np
import pandas as pd
import random
from impyute.imputations.cs.averaging_imputations import mean_imputation

class experience_buffer():
    def __init__(self, buffer_size=5000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def fillWithMean(df):
    if all(df.isnull()): return df.fillna(0.0)
    return df.fillna(df.mean())

def fillWithEWMean(df):
    if all(df.isnull()): return df.fillna(0.0)
    return df.fillna(df.ewm(alpha=0.9, ignore_na=True, adjust=False).mean())

def forwardFill(df):
    if all(df.isnull()): return df.fillna(0.0)
    return df.ffill()

buffer = experience_buffer()

env_name = 'CartPole-v1'
environment= gym.make(env_name)

mcar_rate = 0.5
state_size = 4
s = environment.reset()
state_rollout = []

for i in range(buffer.buffer_size):
    a = environment.action_space.sample()
    s1, r, d, info = environment.step(a)
    if mcar_rate > 0:
        mask_s = np.random.choice([np.nan, 1.0], size=(state_size), p=[mcar_rate, 1 - mcar_rate])
        mask_s1 = np.random.choice([np.nan, 1.0], size=(state_size), p=[mcar_rate, 1 - mcar_rate])
        s = mask_s * s
        state_rollout.append(s)
        if len(state_rollout) == 1:
            s_imputed = pd.DataFrame(state_rollout).fillna(0.0).as_matrix()
        else:
            s_imputed = fillWithEWMean(pd.DataFrame(state_rollout)).as_matrix()[-1]
        s1 = mask_s1 * s1
        rollout = [s_imputed, s1]
        s1_imputed = fillWithEWMean(pd.DataFrame(rollout)).as_matrix()[-1]
        buffer.add(np.reshape(np.array([s_imputed, a, r, s1_imputed, d]),(1,state_size)))
    else:
        buffer.add(np.reshape(np.array([s,a,r,s1,d*1]),[1,5]))
    if d:
        s = environment.reset()
        state_rollout = []
    else: s = s1

x = 1



'''


class ModelLearner:
    def __init__(self, observation_space, action_space, data_size=50000, epochs=50, learning_rate=.001,
                 tmodel_dim_multipliers=(6, 6), tmodel_activations=('relu', 'sigmoid'), sequence_length=1,
                 mcar_rate=0.0, imputer = Imputer()):

        # get size of state and action from environment
        self.state_size = sum(observation_space.shape)
        self.action_size = 1
        self.action_num = 0
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_num = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            self.action_size = sum(action_space.shape)
        else:
            raise ValueError("The action_space is of type: {} - which is not supported!".format(type(action_space)))

        # These are hyper parameters
        self.learning_rate = learning_rate
        self.net_train_epochs = epochs
        self.useRNN = sequence_length > 1
        self.sequence_length = sequence_length
        self.mcar_rate = mcar_rate
        self.inputer = imputer

        # create replay memory using deque
        self.data_size = data_size
        self.memory = deque(maxlen=self.data_size)

        # create main model and target model
        if self.useRNN:
            self.tmodel = self.build_recurrent_regression_model(self.state_size + self.action_size, self.state_size,
                                                                lr=learning_rate,
                                                                dim_multipliers=tmodel_dim_multipliers,
                                                                activations=tmodel_activations)
            self.seq_mem = deque(maxlen=self.sequence_length)
        else:
            self.tmodel = self.build_regression_model(self.state_size + self.action_size, self.state_size,
                                                      lr=learning_rate,
                                                      dim_multipliers=tmodel_dim_multipliers,
                                                      activations=tmodel_activations)
        self.rmodel = self.build_regression_model(self.state_size, 1, lr=learning_rate,
                                                  dim_multipliers=(4, 4),
                                                  activations=('sigmoid', 'sigmoid'))
        self.dmodel = self.build_dmodel(self.state_size)

        self.Ttensorboard = []  # [TensorBoard(log_dir='./logs/Tlearn/{}'.format(time()))]
        self.Rtensorboard = []  # [TensorBoard(log_dir='./logs/Rlearn/{}'.format(time()))]
        self.Dtensorboard = []  # [TensorBoard(log_dir='./logs/Dlearn/{}'.format(time()))]

    # get action from model using random policy
    def get_action(self, state, environment):
        return environment.action_space.sample()


    def refill_mem(self, environment):
        state = environment.reset()
        self.memory.clear()
        for i in range(self.data_size):
            # get action for the current state and go one step in environment
            action = self.get_action(state, environment)
            next_state, reward, done, info = environment.step(action)
            # save the sample <s, a, r, s'> to the replay memory
            self.memory.append(np.hstack((state, action, reward, next_state, done * 1)))
            if done:
                state = environment.reset()
            else:
                state = next_state


    def make_mem_partial_obs(self, memory):
        masks_states = np.random.choice([np.nan, 1.0], size=(len(memory), self.state_size),
                                        p=[self.partial_obs_rate, 1 - self.partial_obs_rate])
        masks_next_states = np.random.choice([np.nan, 1.0], size=(len(memory), self.state_size),
                                             p=[self.partial_obs_rate, 1 - self.partial_obs_rate])
        memory[:, self.state_size:self.state_size+1] = memory[:, self.state_size:self.state_size+1]
        for i in range(len(memory)):
            if i == 0 or memory[i - 1, -1]:
                memory[i, :self.state_size] = masks_states[i] * memory[i, :self.state_size]
                memory[i, :self.state_size][np.isnan(memory[i, :self.state_size])] = 0
                memory[i, self.state_size + self.action_size + 1:-1] = masks_next_states[i] * memory[i, self.state_size + self.action_size + 1:-1]
                for j in range(self.state_size):
                    if np.isnan(memory[i, self.state_size + self.action_size + 1:-1][j]):
                        memory[i, self.state_size + self.action_size + 1:-1][j] = memory[i, :self.state_size][j]
            else:
                memory[i, :self.state_size] = memory[i - 1, self.state_size + self.action_size + 1:-1]
                memory[i, self.state_size + self.action_size + 1:-1] = masks_next_states[i] * memory[i, self.state_size + self.action_size + 1:-1]
                for j in range(self.state_size):
                    if np.isnan(memory[i, self.state_size + self.action_size + 1:-1][j]):
                        memory[i, self.state_size + self.action_size + 1:-1][j] = memory[i, :self.state_size][j]



    def corrupt_mem(self, memory):
        masks_states = np.random.choice([np.nan, 1.0], size=(len(memory), self.state_size),
                                        p=[self.mcar_rate, 1 - self.mcar_rate])
        masks_next_states = np.random.choice([np.nan, 1.0], size=(len(memory), self.state_size),
                                             p=[self.mcar_rate, 1 - self.mcar_rate])
        memory[:, self.state_size:self.state_size+1] = memory[:, self.state_size:self.state_size+1]
        for i in range(len(memory)):
            if i == 0 or memory[i - 1, -1]:
                memory[i, :self.state_size] = masks_states[i] * memory[i, :self.state_size]
                memory[i, self.state_size + self.action_size + 1:-1] = masks_next_states[i] * memory[i, self.state_size + self.action_size + 1:-1]
            else:
                memory[i, :self.state_size] = memory[i - 1, self.state_size + self.action_size + 1:-1]
                memory[i, self.state_size + self.action_size + 1:-1] = masks_next_states[i] * memory[i, self.state_size + self.action_size + 1:-1]


    def impute_mem(self, memory):
        imputer = Imputer()
        imputed_memory = imputer.fit_transform(memory)
        return imputed_memory

    def setup_batch_for_RNN(self, batch):
        batch_size = batch.shape[0]
        array_size = batch_size - self.sequence_length
        x_seq = np.empty((array_size, self.sequence_length, self.state_size + self.action_size))
        y_seq = np.empty((array_size, self.state_size))
        actual_size = 0
        for jj in range(array_size):
            if not batch[jj:jj + self.sequence_length - 1, -1].any():
                x_seq[actual_size, ...] = batch[np.newaxis,
                                          jj:jj + self.sequence_length,
                                          :self.state_size + self.action_size]
                y_seq[actual_size, ...] = batch[np.newaxis,
                                          jj + self.sequence_length - 1,
                                          -self.state_size - 1:-1]
                actual_size += 1
        x_seq = np.resize(x_seq, (actual_size, self.sequence_length, self.state_size + self.action_size))
        y_seq = np.resize(y_seq, (actual_size, self.state_size))
        return x_seq, y_seq


    # approximate Transition function
    # state and action is input and successor state is output
    def build_regression_model(self,
                               input_dim,
                               output_dim,
                               dim_multipliers=(6, 4),
                               activations=('sigmoid', 'sigmoid'),
                               lr=.001):
        model = Sequential()
        model.add(Dense(self.state_size * dim_multipliers[0], input_dim=input_dim, activation=activations[0]))
        for i in range(len(dim_multipliers) - 1):
            model.add(Dense(self.state_size * dim_multipliers[i + 1],
                            activation=activations[min(i + 1, len(activations) - 1)]))
        model.add(Dense(output_dim, activation='linear'))
        if self.mcar_rate > 0:
            model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['accuracy'])
        model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['accuracy'])
        # model.summary()
        return model

    def build_recurrent_regression_model(self,
                                         input_dim,
                                         output_dim,
                                         dim_multipliers=(6, 4),
                                         activations=('sigmoid', 'sigmoid'),
                                         lr=.001):
        num_hlayers = len(dim_multipliers)
        model = Sequential()
        model.add(LSTM(self.state_size * dim_multipliers[0],
                       input_shape=(None, input_dim),
                       activation=activations[0],
                       return_sequences=num_hlayers is not 1
                       ))
        for i in range(num_hlayers - 1):
            model.add(LSTM(self.state_size * dim_multipliers[i + 1],
                           activation=activations[min(i + 1, len(activations) - 1)],
                           return_sequences=i is not num_hlayers - 2))
            # stacked LSTMs need to return a sequence
        model.add(Dense(output_dim, activation='relu'))
        if self.mcar_rate > 0:
            model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['accuracy'])
        model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['accuracy'])
        # model.summary()
        return model

    # approximate Done value
    # state is input and reward is output
    def build_dmodel(self,
                     input_dim,
                     dim_multipliers=(4, 4),
                     activations=('relu', 'relu'),
                     lr=.001):
        model = Sequential()
        model.add(Dense(self.state_size * dim_multipliers[0], input_dim=input_dim, activation=activations[0]))
        for i in range(len(dim_multipliers) - 1):
            model.add(Dense(self.state_size * dim_multipliers[i + 1], activation=activations[i + 1]))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
        # model.summary()
        return model

    # pick samples randomly from replay memory (with batch_size)
    def train_models(self, minibatch_size=32):
        memory_arr = np.array(self.memory)
        if self.mcar_rate > 0:
            self.corrupt_mem(memory_arr)
            imputed_memory = self.impute_mem(memory_arr)


        if self.useRNN:
            batch_size = len(memory_arr)
            minibatch_size = min(minibatch_size, batch_size)
            t_x, t_y = self.setup_batch_for_RNN(imputed_memory)
            self.tmodel.fit(t_x, t_y,
                            batch_size=minibatch_size,
                            epochs=self.net_train_epochs,
                            validation_split=0.1,
                            callbacks=self.Ttensorboard, verbose=1)
        else:
            batch_size = len(memory_arr)
            minibatch_size = min(minibatch_size, batch_size)
            # batch = random.sample(list(imputed_memory), minibatch_size)
            # batch = np.array(batch)
            batch = imputed_memory
            t_x = batch[:, :self.state_size + self.action_size]
            t_y = batch[:, -self.state_size - 1:-1]
            self.tmodel.fit(t_x, t_y,
                            batch_size=minibatch_size,
                            epochs=self.net_train_epochs * 100000,
                            validation_split=0.1,
                            callbacks=self.Ttensorboard, verbose=1)

            self.rmodel.fit(batch[:, :self.state_size],
                            batch[:, self.state_size + self.action_size],
                            batch_size=minibatch_size,
                            epochs=self.net_train_epochs,
                            validation_split=0.1,
                            callbacks=self.Rtensorboard, verbose=0)

            self.dmodel.fit(batch[:, :self.state_size],
                            batch[:, -1],
                            batch_size=minibatch_size,
                            epochs=self.net_train_epochs,
                            validation_split=0.1,
                            callbacks=self.Dtensorboard, verbose=0)

    def step(self, state, action):
        batch_size = 1
        state = np.reshape(state, [1, self.state_size])
        action = np.reshape(action, [1, self.action_size])

        if self.useRNN:
            self.seq_mem.append(np.hstack((state, action)))
            if len(self.seq_mem) is self.sequence_length:
                seq = np.array(self.seq_mem)
                next_state = self.tmodel.predict(np.rollaxis(seq, 1), batch_size)
            else:
                next_state = state
        else:
            next_state = self.tmodel.predict(np.hstack((state, action)), batch_size)
        reward = self.rmodel.predict(state, batch_size)
        done = self.dmodel.predict(state, batch_size)

        return next_state[0], float(reward[0, 0]), bool(done[0, 0] > .8)
        # TODO how sure do we want to be about being done? 80%? 90?
        # TODO yah to force the type to be the same as in gym environment (flatten?)

    def run(self, environment, rounds=1):
        for e in range(rounds):
            self.refill_mem(environment)
            self.train_models()

    def evaluate(self, environment, num_steps=5000, do_plots=False):
        states, nexts_real, nexts_pred, rewards_real, rewards_pred, dones_real, dones_pred = [], [], [], [], [], [], []
        state = environment.reset()
        states.append(state)

        for episode_number in range(num_steps):
            action = self.get_action(state, environment)

            next_state, reward, done = self.step(state, action)

            nexts_pred.append(next_state)
            rewards_pred.append(reward)
            dones_pred.append(done)

            next_state_real, reward_real, done_real, info_real = environment.step(action)

            nexts_real.append(next_state_real)
            rewards_real.append(reward_real)
            dones_real.append(done_real)

            state = next_state_real
            if done_real:
                state = environment.reset()

        p = np.asarray(nexts_pred)
        r = np.asarray(nexts_real)
        if do_plots:
            for i in range(min(self.state_size, 12)):
                plt.figure(i)
                plt.plot(p[:, i])
                plt.plot(r[:, i])
            plt.figure(self.state_size + 1)
            plt.plot(rewards_pred)
            plt.plot(rewards_real)
            plt.figure(self.state_size + 2)
            plt.plot(dones_pred)
            plt.plot(dones_real)
            plt.show()

        mse = ((p - r) ** 2).mean()
        return mse

def forwardFill(df):
    if all(df.isnull()): return df.fillna(0.0)
    return df.ffill()

def fillWithMean(df):
    if all(df.isnull()): return df.fillna(0.0)
    return df.fillna(df.mean())

def fillWithEWMean(df):
    if all(df.isnull()): return df.fillna(0.0)
    return df.fillna(df.ewm(alpha=0.9, ignore_na=True, adjust=False).mean())




if __name__ == "__main__":
    # ['Ant-v1', 'LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']:"Pong-ram-v4"
    for env_name in ['CartPole-v1']:
        env = gym.make(env_name)

        canary = ModelLearner(env.observation_space, env.action_space, mcar_rate=0.75, sequence_length=1)
        canary.run(env, rounds=1)

        print('MSE: {}'.format(canary.evaluate(env)))
