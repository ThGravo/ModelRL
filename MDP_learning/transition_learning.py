import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import TensorBoard


class ModelLearner:
    def __init__(self, state_size, action_size):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters
        self.learning_rate = 0.001
        self.batch_size = 8000
        self.net_train_epochs = 16
        # create replay memory using deque
        self.mem_size = self.batch_size
        self.memory = deque(maxlen=self.mem_size)

        # create main model and target model
        self.tmodel = self.build_tmodel()
        self.rmodel = self.build_rmodel()
        self.dmodel = self.build_dmodel()

        self.Ttensorboard = TensorBoard(log_dir='./logs/Tlearn/{}'.format(time()))
        self.Rtensorboard = TensorBoard(log_dir='./logs/Rlearn/{}'.format(time()))
        self.Dtensorboard = TensorBoard(log_dir='./logs/Dlearn/{}'.format(time()))

    # approximate Transition function
    # state and action is input and successor state is output
    def build_tmodel(self):
        tmodel = Sequential()
        tmodel.add(Dense(24, input_dim=self.state_size + 1, activation='relu',
                         kernel_initializer='he_uniform'))
        tmodel.add(Dense(24, activation='relu',
                         kernel_initializer='he_uniform'))
        tmodel.add(Dense(self.state_size, activation='linear',
                         kernel_initializer='he_uniform'))
        tmodel.summary()
        tmodel.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return tmodel

    # approximate Reward function
    # state is input and reward is output
    def build_rmodel(self):
        rmodel = Sequential()
        rmodel.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        rmodel.add(Dense(24, activation='relu',
                         kernel_initializer='he_uniform'))
        rmodel.add(Dense(1, activation='linear',
                         kernel_initializer='he_uniform'))
        rmodel.summary()
        rmodel.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return rmodel

    # approximate Done value
    # state is input and reward is output
    def build_dmodel(self):
        dmodel = Sequential()
        dmodel.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        dmodel.add(Dense(24, activation='relu',
                         kernel_initializer='he_uniform'))
        dmodel.add(Dense(1, activation='sigmoid',
                         kernel_initializer='he_uniform'))
        dmodel.summary()
        dmodel.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return dmodel

    # get action from model using random policy
    def get_action(self, state):
        return random.randrange(self.action_size)

    # pick samples randomly from replay memory (with batch_size)
    def train_models(self, minibatch_size=32):
        batch_size = min(self.batch_size, len(self.memory))
        minibatch_size = min(minibatch_size, batch_size)

        mini_batch = list(self.memory)

        update_input = np.zeros(
            (batch_size, self.state_size + 1))  # TODO action repesentation & corresponding network architecture
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i][:self.state_size] = mini_batch[i][0]
            update_input[i][-1] = mini_batch[i][1]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        # and do the model fit
        self.tmodel.fit(update_input, update_target, batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        verbose=0,
                        validation_split=0.1, callbacks=[self.Ttensorboard]
                        )

        # TODO Currently predicts reward based on state input data.
        #  Should we consider making reward predictions action-dependent too?
        self.rmodel.fit(update_input[:, :-1], reward, batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        verbose=0,
                        validation_split=0.1, callbacks=[self.Rtensorboard]
                        )

        self.dmodel.fit(update_input[:, :-1], done, batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        verbose=0,
                        validation_split=0.1, callbacks=[self.Dtensorboard]
                        )

    def step(self, state, action):
        batch_size = 1

        # TODO definitely not the most pythonic way 
        x = np.zeros((batch_size, self.state_size + 1))
        x[0][:self.state_size] = np.reshape(state, [1, self.state_size])
        x[0][-1] = action

        next_state = self.tmodel.predict(x, batch_size)
        reward = self.rmodel.predict(x[:, :-1], batch_size)
        done = self.dmodel.predict(x[:, :-1], batch_size)

        return next_state[0], float(reward[0, 0]), bool(done[0, 0] > .5)
        # TODO yah to force the type to be the same as in gym environment

    def fill_mem(self, environment):
        state = environment.reset()

        for i in range(self.mem_size):
            # get action for the current state and go one step in environment
            action = self.get_action(state)
            next_state, reward, done, info = environment.step(action)
            print(info, reward)

            # save the sample <s, a, r, s'> to the replay memory
            self.memory.append((np.reshape(state, [1, self.state_size]),
                                action,
                                reward,
                                np.reshape(next_state, [1, self.state_size]),
                                done*1))

            if done and np.random.rand() <= .5:  # TODO super hacky way to get 0 rewards in cartpole
                state = environment.reset()
            else:
                state = next_state


if __name__ == "__main__":
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    scores, episodes, val_accs, episodes_val = [], [], [], []

    canary = ModelLearner(state_size, action_size)

    for e in range(2):
        print('Filling Replay Memory...')
        canary.fill_mem(env)
        print('Training...')
        canary.train_models()
        canary.memory.clear()

    '''Evaluation'''
    episode_number = 1

    states, nexts_real, nexts_pred, rewards_real, rewards_pred, dones_real, dones_pred = [], [], [], [], [], [], []

    state = env.reset()

    states.append(state)

    while episode_number <= 150:
        action = 1 if np.random.uniform() < .5 else 0

        next_state, reward, done = canary.step(state, action)

        nexts_pred.append(next_state)
        rewards_pred.append(reward)
        dones_pred.append(done)

        next_state_real, reward_real, done_real, info_real = env.step(action)

        nexts_real.append(next_state_real)
        rewards_real.append(reward_real)
        dones_real.append(done_real)

        state = next_state_real
        if done_real:
            env.reset()
        episode_number += 1

    p = np.asarray(nexts_pred)
    r = np.asarray(nexts_real)
    """"""
    for i in range(4):
        plt.figure(i)
        plt.plot(p[:, i])
        plt.plot(r[:, i])
    plt.figure(4)
    plt.plot(rewards_pred)
    plt.plot(rewards_real)
    plt.figure(5)
    plt.plot(dones_pred)
    plt.plot(dones_real)

    #plt.tight_layout()
    plt.show()

















