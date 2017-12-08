import tensorflow as tf
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import TensorBoard
import gym
import random
import numpy as np
import matplotlib
# matplotlib.use('GTK3Cairo', warn=False, force=True)
import matplotlib.pyplot as plt

'''
GRID SEARCH RESULT
Best parameter set was 
{'learning_rate': 0.001, 'tmodel_activations': ('relu', 'sigmoid'), 'tmodel_dim_multipliers': (6, 6)}
'''


class ModelLearner:
    def __init__(self, observation_space, action_space, data_size=250000, epochs=50, learning_rate=.001,
                 tmodel_dim_multipliers=(6, 6), tmodel_activations=('relu', 'sigmoid')):

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
        # create replay memory using deque
        self.data_size = data_size
        self.memory = deque(maxlen=self.data_size)

        # create main model and target model
        self.tmodel = self.build_regression_model(self.state_size + self.action_size, self.state_size, lr=learning_rate,
                                                  dim_multipliers=tmodel_dim_multipliers,
                                                  activations=tmodel_activations)
        self.rmodel = self.build_regression_model(self.state_size, 1, lr=learning_rate,
                                                  dim_multipliers=(4, 4),
                                                  activations=('sigmoid', 'sigmoid'))
        self.dmodel = self.build_dmodel(self.state_size)

        self.Ttensorboard = []  # [TensorBoard(log_dir='./logs/Tlearn/{}'.format(time()))]
        self.Rtensorboard = []  # [TensorBoard(log_dir='./logs/Rlearn/{}'.format(time()))]
        self.Dtensorboard = []  # [TensorBoard(log_dir='./logs/Dlearn/{}'.format(time()))]

    # approximate Transition function
    # state and action is input and successor state is output
    def build_regression_model(self,
                               input_dim,
                               output_dim,
                               dim_multipliers=(6, 4),
                               activations=('relu', 'relu'),
                               lr=.001):
        model = Sequential()
        model.add(Dense(self.state_size * dim_multipliers[0], input_dim=input_dim, activation=activations[0]))
        for i in range(len(dim_multipliers) - 1):
            model.add(Dense(self.state_size * dim_multipliers[i + 1],
                            activation=activations[min(i + 1, len(activations) - 1)]))
        model.add(Dense(output_dim, activation='linear'))
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

    # get action from model using random policy
    def get_action(self, state, environment):
        return environment.action_space.sample()

    # pick samples randomly from replay memory (with batch_size)
    def train_models(self, minibatch_size=32):
        batch_size = len(self.memory)
        minibatch_size = min(minibatch_size, batch_size)

        batch = np.array(self.memory)

        # and do the model fit
        self.tmodel.fit(batch[:, :self.state_size + self.action_size],
                        batch[:, -self.state_size - 1:-1],
                        batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        validation_split=0.1,
                        callbacks=self.Ttensorboard, verbose=1)

        # TODO Currently predicts reward based on state input data.
        #  Should we consider making reward predictions action-dependent too?
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

        next_state = self.tmodel.predict(np.hstack((state, action)), batch_size)
        reward = self.rmodel.predict(state, batch_size)
        done = self.dmodel.predict(state, batch_size)

        return next_state[0], float(reward[0, 0]), bool(done[0, 0] > .8)
        # TODO how sure do we want to be about being done? 80%? 90?
        # TODO yah to force the type to be the same as in gym environment (flatten?)

    def refill_mem(self, environment):
        state = environment.reset()
        self.memory.clear()
        for i in range(self.data_size):
            # get action for the current state and go one step in environment
            action = self.get_action(state, environment)
            next_state, reward, done, info = environment.step(action)

            # save the sample <s, a, r, s'> to the replay memory
            self.memory.append(np.hstack((state, action, reward, next_state, done * 1)))

            if done:  # and np.random.rand() <= .5:  # TODO super hacky way to get 0 rewards in cartpole
                state = environment.reset()
            else:
                state = next_state

    def run(self, environment, rounds=1):
        for e in range(rounds):
            self.refill_mem(environment)
            self.train_models()

    def evaluate(self, environment, do_plots=False):
        states, nexts_real, nexts_pred, rewards_real, rewards_pred, dones_real, dones_pred = [], [], [], [], [], [], []
        state = environment.reset()
        states.append(state)

        for episode_number in range(5000):
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
    for env_name in ['Ant-v1']:  # ['LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']:"Pong-ram-v4"
        env = gym.make(env_name)

        canary = ModelLearner(env.observation_space, env.action_space)
        canary.run(env, rounds=8)

        print('MSE: {}'.format(canary.evaluate(env)))
