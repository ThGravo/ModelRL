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
from sklearn.preprocessing import Imputer, MinMaxScaler
from fancyimpute import KNN, SimpleFill, SoftImpute, MICE, IterativeSVD, NuclearNormMinimization, MatrixFactorization, BiScaler
from MDP_learning.single_agent.preprocessing import standardise_memory, make_mem_partial_obs, setup_batch_for_RNN, impute_missing
from MDP_learning.single_agent.networks import build_regression_model, build_recurrent_regression_model, build_dmodel

class ModelLearner:
    def __init__(self, observation_space, action_space, data_size=5000000, epochs=100, learning_rate=.001,
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

        if self.useRNN:
            self.tmodel = build_recurrent_regression_model(self.state_size + self.action_size, self.state_size,
                                                                lr=learning_rate,
                                                                dim_multipliers=tmodel_dim_multipliers,
                                                                activations=tmodel_activations)
            self.seq_mem = deque(maxlen=self.sequence_length)
        else:
            self.tmodel = build_regression_model(self.state_size + self.action_size, self.state_size,
                                                      lr=learning_rate,
                                                      dim_multipliers=tmodel_dim_multipliers,
                                                      activations=tmodel_activations)
        self.rmodel = build_regression_model(self.state_size, 1, lr=learning_rate,
                                                  dim_multipliers=(4, 4),
                                                  activations=('sigmoid', 'sigmoid'))
        self.dmodel = build_dmodel(self.state_size)

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
            action = self.get_action(state, environment)
            next_state, reward, done, info = environment.step(action)
            self.memory.append(np.hstack((state, action, reward, next_state, done * 1)))
            if done:
                state = environment.reset()
            else:
                state = next_state

    def train_models(self, minibatch_size=512):
        # memory_arr = np.array(random.sample(list(np.array(self.memory)),100000))
        memory_arr = np.array(self.memory)
        if self.partial_obs_rate > 0:
            make_mem_partial_obs(memory_arr,self.state_size,self.partial_obs_rate)
            print("Memory size:")
            print(memory_arr.size)
            print("Proportion of missing values:")
            print(np.isnan(memory_arr).sum() / memory_arr.size)
            impute_missing(memory_arr, self.state_size, MICE)
        else:
            standardise_memory(memory_arr, self.state_size, self.action_size)
        batch_size = len(memory_arr)
        minibatch_size = min(minibatch_size, batch_size)

        if self.useRNN:
            t_x, t_y = setup_batch_for_RNN(memory_arr,self.sequence_length,self.state_size,self.action_size)
        else:
            t_x = memory_arr[:, :self.state_size + self.action_size]
            t_y = memory_arr[:, -self.state_size - 1:-1] - memory_arr[:, :self.state_size]

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


if __name__ == "__main__":
    # ['Ant-v1', 'LunarLander-v2', 'BipedalWalker-v2', FrozenLake8x8-v0, 'MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']:"Pong-ram-v4"
    for env_name in ['Swimmer-v1', 'Hopper-v1']:
        env = gym.make(env_name)
        print(env.observation_space)
        canary = ModelLearner(env.observation_space, env.action_space, partial_obs_rate=0.0, sequence_length=1)
        canary.refill_mem(env)
        memory_arr = np.array(canary.memory)
        print("Saving memory")
        np.save('/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'FULL',
                memory_arr)
        for rate in [0.05, 0.1]:
            make_mem_partial_obs(memory_arr,canary.state_size,rate)
            print("Saving corrupted memory")
            np.save(
                '/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'CORRUPT' + str(rate), memory_arr)
            impute_missing(memory_arr,canary.state_size,MICE)
            print("Saving imputed memory")
            np.save(
                '/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'IMPUTED' + str(rate) + 'round' + str(
                    round), memory_arr)



       