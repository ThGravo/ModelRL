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
    def __init__(self, state_size, num_discrete_actions, action_size=1):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.action_num = num_discrete_actions

        # These are hyper parameters
        self.learning_rate = 0.001
        self.batch_size = 16000
        self.net_train_epochs = 32
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
        tmodel.add(Dense(self.state_size * 6, input_dim=self.state_size + self.action_size, activation='relu',
                         kernel_initializer='he_uniform'))
        tmodel.add(Dense(self.state_size * 4, activation='relu',
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
        rmodel.add(Dense(self.state_size * 4, input_dim=self.state_size, activation='sigmoid',
                         kernel_initializer='he_uniform'))
        rmodel.add(Dense(self.state_size * 4, activation='sigmoid',
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
        dmodel.add(Dense(self.state_size * 4, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        dmodel.add(Dense(self.state_size * 4, activation='relu',
                         kernel_initializer='he_uniform'))
        dmodel.add(Dense(1, activation='sigmoid',
                         kernel_initializer='he_uniform'))
        dmodel.summary()
        dmodel.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return dmodel

    # get action from model using random policy
    def get_action(self, state):
        return random.randrange(self.action_num)

    # pick samples randomly from replay memory (with batch_size)
    def train_models(self, minibatch_size=32):
        batch_size = len(self.memory)
        minibatch_size = min(minibatch_size, batch_size)

        batch = np.array(self.memory)

        # and do the model fit
        self.tmodel.fit(batch[:, :self.state_size + self.action_size],
                        batch[:, -self.state_size-1:-1],
                        batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        validation_split=0.1,
                        callbacks=[self.Ttensorboard]
                        )

        # TODO Currently predicts reward based on state input data.
        #  Should we consider making reward predictions action-dependent too?
        self.rmodel.fit(batch[:, :self.state_size],
                        batch[:, self.state_size + self.action_size],
                        batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        validation_split=0.1,
                        callbacks=[self.Rtensorboard]
                        )

        self.dmodel.fit(batch[:, :self.state_size],
                        batch[:, -1],
                        batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        validation_split=0.1,
                        callbacks=[self.Dtensorboard]
                        )

    def step(self, state, action):
        batch_size = 1
        state = np.reshape(state, [1, self.state_size])
        action = np.reshape(action, [1, self.action_size])

        next_state = self.tmodel.predict(np.hstack((state, action)), batch_size)
        reward = self.rmodel.predict(state, batch_size)
        done = self.dmodel.predict(state, batch_size)

        return next_state[0], float(reward[0, 0]), bool(done[0, 0] > .8)
        # TODO how sure do we want to be about being done? 80%? 90?
        # TODO yah to force the type to be the same as in gym environment

    def fill_mem(self, environment):
        state = environment.reset()

        for i in range(self.mem_size):
            # get action for the current state and go one step in environment
            action = self.get_action(state)
            next_state, reward, done, info = environment.step(action)
            # print(info, reward)

            # save the sample <s, a, r, s'> to the replay memory
            self.memory.append(np.hstack(
                (state, action, reward, next_state, done * 1)
            ))

            if done:  # and np.random.rand() <= .5:  # TODO super hacky way to get 0 rewards in cartpole
                state = environment.reset()
            else:
                state = next_state


if __name__ == "__main__":
    env_name = 'CartPole-v1'
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    # get size of state and action from environment
    state_size = sum(env.observation_space.shape)
    num_discrete_actions = env.action_space.n

    canary = ModelLearner(state_size, num_discrete_actions)

    for e in range(16):
        print('Filling Replay Memory...')
        canary.fill_mem(env)
        print('Training...')
        canary.train_models()
        canary.memory.clear()

    '''Evaluation'''
    episode_number = 1
    scores, episodes, val_accs, episodes_val = [], [], [], []

    states, nexts_real, nexts_pred, rewards_real, rewards_pred, dones_real, dones_pred = [], [], [], [], [], [], []

    state = env.reset()

    states.append(state)

    while episode_number <= 500:
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
    for i in range(state_size):
        plt.figure(i)
        plt.plot(p[:, i])
        plt.plot(r[:, i])
    plt.figure(state_size+1)
    plt.plot(rewards_pred)
    plt.plot(rewards_real)
    plt.figure(state_size+2)
    plt.plot(dones_pred)
    plt.plot(dones_real)

    # plt.tight_layout()
    plt.show()
