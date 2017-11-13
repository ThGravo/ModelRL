import gym
import random
import numpy as np
from time import time
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import TensorBoard

EPISODES = 1


class TLearner:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters
        self.learning_rate = 0.0001
        self.batch_size = 500
        self.net_train_epochs = 128
        # create replay memory using deque
        self.mem_size = self.batch_size
        self.memory = deque(maxlen=self.mem_size)

        # create main model and target model
        self.tmodel = self.build_tmodel()
        self.rmodel = self.build_rmodel()
        self.dmodel = self.build_dmodel()

        self.Ttensorboard = TensorBoard(log_dir='./logs/Tlearn/{}'.format(time()),
                                        histogram_freq=1,
                                        write_graph=True,
                                        write_images=False)
        self.Rtensorboard = TensorBoard(log_dir='./logs/Rlearn/{}'.format(time()),
                                        histogram_freq=1,
                                        write_graph=True,
                                        write_images=False)
        self.Dtensorboard = TensorBoard(log_dir='./logs/Dlearn/{}'.format(time()),
                                        histogram_freq=1,
                                        write_graph=True,
                                        write_images=False)

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
        # dmodel.add(Dense(24, activation='relu',
        #                 kernel_initializer='he_uniform'))
        dmodel.add(Dense(1, activation='sigmoid',
                         kernel_initializer='he_uniform'))
        dmodel.summary()
        dmodel.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return dmodel

    # get action from model using random policy
    def get_action(self, state):
        return random.randrange(self.action_size)

    # pick samples randomly from replay memory (with batch_size)
    def train_models(self, minibatch_size):
        batch_size = min(self.batch_size, len(self.memory))
        minibatch_size = min(minibatch_size, batch_size)

        mini_batch = random.sample(self.memory, batch_size)

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

        # and do the model fit!
        self.tmodel.fit(update_input, update_target, batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        verbose=0,
                        validation_split=0.1,
                        callbacks=[self.Ttensorboard]                        )

        self.rmodel.fit(update_input[:, :-1], reward, batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        verbose=0,
                        validation_split=0.1,
                        callbacks=[self.Rtensorboard])
        # TODO Currently predicts reward based on state input data.
        #  Should we consider making reward predictions action-dependent too?

        self.dmodel.fit(update_input[:, :-1], done, batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        verbose=0,
                        validation_split=0.1,
                        callbacks=[self.Dtensorboard])

    def fill_mem(self, environment):
        state = environment.reset()
        state = np.reshape(state, [1, state_size])

        for i in range(self.mem_size):
            if self.render:
                environment.render()

            # get action for the current state and go one step in environment
            action = self.get_action(state)
            next_state, reward, done, info = environment.step(action)
            print(reward)
            # save the sample <s, a, r, s'> to the replay memory
            self.memory.append((state, action, reward, np.reshape(next_state, [1, state_size]), done))
            if done:
                state = environment.reset()
            else:
                state = next_state


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = TLearner(state_size, action_size)

    scores, episodes, val_accs, episodes_val = [], [], [], []

    for e in range(EPISODES):
        print('Filling Replay Memory...')
        agent.fill_mem(env)
        print('Training...')
        agent.train_models(10)
        agent.memory.clear()
