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
from fancyimpute import KNN, SimpleFill, SoftImpute, MICE, IterativeSVD

'''
GRID SEARCH RESULT
Best parameter set was 
{'learning_rate': 0.001, 'tmodel_activations': ('relu', 'sigmoid'), 'tmodel_dim_multipliers': (6, 6)}
'''


class ModelLearner:
    def __init__(self, observation_space, action_space, data_size=75000, epochs=4000, learning_rate=.001,
                 tmodel_dim_multipliers=(1, 1), tmodel_activations=('relu', 'relu'), sequence_length=1,
                 partial_obs_rate=0.0):

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
        self.partial_obs_rate = partial_obs_rate

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
                memory[i, self.state_size + self.action_size + 1:-1] = masks_next_states[i] * memory[i, self.state_size + self.action_size + 1:-1]
            else:
                memory[i, :self.state_size] = memory[i - 1, self.state_size + self.action_size + 1:-1]
                memory[i, self.state_size + self.action_size + 1:-1] = masks_next_states[i] * memory[i, self.state_size + self.action_size + 1:-1]


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
        model.add(Dense(1000, input_dim=input_dim, activation=activations[0]))
        for i in range(len(dim_multipliers) - 1):
            model.add(Dense(1000,
                            activation=activations[min(i + 1, len(activations) - 1)]))
        model.add(Dense(output_dim, activation='linear'))
        if self.partial_obs_rate > 0:
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
        if self.partial_obs_rate > 0:
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
    def train_models(self, minibatch_size=512):
        #file = "memoryV2.npy"
        #np.save(file, np.array(self.memory))
        memory_arr = np.array(random.sample(list(np.array(self.memory)),100000))

        if self.partial_obs_rate > 0:
            self.make_mem_partial_obs(memory_arr)
            file1 = "memoryBWcorrupted.npy"
            np.save(file1, memory_arr)
            print("Memory size:")
            print(memory_arr.size)
            print("Proportion of missing values:")
            print(np.isnan(memory_arr).sum() / memory_arr.size)
            #memory_train = np.array([exp for exp in memory_arr if not np.isnan(exp[-self.state_size - 1:-1]).any()])
            #imputer = Imputer()
            #memory_final = imputer.fit_transform(memory_train)
            memory_final = SoftImpute().complete(memory_arr)
            file2 = "memoryBWimputedSoft.npy"
            np.save(file2, memory_final)
        else:
            memory_final = memory_arr
        if self.useRNN:
            batch_size = len(memory_final)
            minibatch_size = min(minibatch_size, batch_size)
            t_x, t_y = self.setup_batch_for_RNN(memory_final)
            self.tmodel.fit(t_x, t_y,
                            batch_size=minibatch_size,
                            epochs=self.net_train_epochs,
                            validation_split=0.1,
                            callbacks=self.Ttensorboard, verbose=1)
        else:
            batch_size = len(memory_arr)
            minibatch_size = min(minibatch_size, batch_size)
            # batch = random.sample(list(memory_final), minibatch_size)
            # batch = np.array(batch)
            # batch = memory_arr
            t_x = memory_final[:, :self.state_size + self.action_size]
            t_y = memory_final[:, -self.state_size - 1:-1] - memory_final[:, :self.state_size]
            self.tmodel.fit(t_x, t_y,
                            batch_size=minibatch_size,
                            epochs=self.net_train_epochs,
                            validation_split=0.1,
                            callbacks=self.Ttensorboard, verbose=1)
            '''
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
            '''
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


def weighted_mean_squared_error(y_true, y_pred):
    total = K.backend.sum(K.backend.square(K.backend.abs(K.backend.sign(y_true)) * (y_pred - y_true)), axis=-1)
    count = K.backend.sum(K.backend.abs(K.backend.sign(y_true)))
    return total / count


if __name__ == "__main__":
    # ['Ant-v1', 'LunarLander-v2', 'BipedalWalker-v2', FrozenLake8x8-v0, 'MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']:"Pong-ram-v4"
    for env_name in ['BipedalWalker-v2']:
        env = gym.make(env_name)
        print(env.observation_space)
        canary = ModelLearner(env.observation_space, env.action_space, partial_obs_rate=0.0, sequence_length=1)
        #canary.refill_mem(env)
        print("loading memory")
        canary.memory = np.load('/home/aocc/code/DL/MDP_learning/save_memory/first_20mill/BipedalWalker-v2FULL.npy')
        print(canary.memory.shape)
        t_x = canary.memory[:, :canary.state_size + canary.action_size]
        t_y = canary.memory[:, -canary.state_size - 1:-1] - canary.memory[:, :canary.state_size]
        canary.tmodel.fit(t_x, t_y,
                        batch_size=512,
                        epochs=canary.net_train_epochs,
                        validation_split=0.1,
                        callbacks=canary.Ttensorboard, verbose=1)
        #canary.run(env, rounds=1)

        print('MSE: {}'.format(canary.evaluate(env)))
