from MDP_learning.multi_agent import make_env

from collections import deque
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.models import Sequential
import random
import numpy as np
# matplotlib.use('GTK3Cairo', warn=False, force=True)
import matplotlib.pyplot as plt

class ModelLearner:
    def __init__(self, env, data_size=3000, epochs=4, learning_rate=.001,
                 tmodel_dim_multipliers=(6, 6), tmodel_activations=('relu', 'relu'), sequence_length=1):

        self.env = env
        self.n = self.env.n # number of agents
        # get size of state and action from environment
        self.state_size = []
        self.action_num = []
        self.action_size = 5
        for i in range(self.n):
            self.state_size.append(sum(self.env.observation_space[i].shape))
            self.action_num.append(env.action_space[i].n)  # assuming discrete action spaces
        self.learning_rate = learning_rate
        self.net_train_epochs = epochs
        self.data_size = data_size
        self.memory = deque(maxlen=self.data_size)
        self.useRNN = sequence_length > 1
        self.sequence_length = sequence_length
        self.tmodel = []
        for i in range(self.n):
            if self.useRNN:
                model = self.build_recurrent_regression_model(self.state_size[i] + self.n* self.action_size, self.state_size[i], lr=learning_rate,
                                                          dim_multipliers=tmodel_dim_multipliers,
                                                          activations=tmodel_activations)
                self.seq_mem = deque(maxlen=self.sequence_length)
                self.tmodel.append(model)
            else:
                model = self.build_regression_model(self.state_size[i] + self.n* self.action_size, self.state_size[i], lr=learning_rate,
                                                          dim_multipliers=tmodel_dim_multipliers,
                                                          activations=tmodel_activations)
                self.tmodel.append(model)
        '''
        self.rmodel = self.build_regression_model(100, 1, lr=learning_rate,
                                                  dim_multipliers=(4, 4),
                                                  activations=('sigmoid', 'sigmoid'))
        self.dmodel = self.build_dmodel(100)
        '''
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
        model.add(Dense(500, input_dim=input_dim, activation=activations[0]))
        for i in range(len(dim_multipliers) - 1):
            model.add(Dense(500,
                            activation=activations[min(i + 1, len(activations) - 1)]))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['accuracy'])
        model.summary()
        return model

    def build_recurrent_regression_model(self,
                                         input_dim,
                                         output_dim,
                                         dim_multipliers=(6, 4),
                                         activations=('sigmoid', 'sigmoid'),
                                         lr=.001):
        num_hlayers = len(dim_multipliers)
        model = Sequential()
        model.add(LSTM(1000,
                       input_shape=(None, input_dim),
                       activation=activations[0],
                       return_sequences=num_hlayers is not 1
                       ))
        for i in range(num_hlayers - 1):
            model.add(LSTM(1000,
                           activation=activations[min(i + 1, len(activations) - 1)],
                           return_sequences=i is not num_hlayers - 2))
            # stacked LSTMs need to return a sequence
        model.add(Dense(output_dim, activation='relu'))
        model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['accuracy'])
        model.summary()
        return model



    # approximate Done value
    # state is input and reward is output
    def build_dmodel(self,
                     input_dim,
                     dim_multipliers=(4, 4),
                     activations=('relu', 'relu'),
                     lr=.001):
        model = Sequential()
        model.add(Dense(20 * dim_multipliers[0], input_dim=input_dim, activation=activations[0]))
        for i in range(len(dim_multipliers) - 1):
            model.add(Dense(20 * dim_multipliers[i + 1], activation=activations[i + 1]))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
        # model.summary()
        return model

    # get action from model using random policy
    def get_action(self, state, environment):
        actions = []
        for i in range(self.n):
            act = np.array([random.uniform(-2,2) for j in range(env.action_space[i].n)])
            actions.append(act)
        return actions

    # pick samples randomly from replay memory (with batch_size)
    def train_models(self, minibatch_size=32):
        batch_size = len(self.memory)
        minibatch_size = min(minibatch_size, batch_size)
        batch = np.array(self.memory)
        if self.useRNN:
            t_x, t_y = self.setup_batch_for_RNN(batch)
            self.tmodel.fit(t_x, t_y,
                            batch_size=minibatch_size,
                            epochs=self.net_train_epochs,
                            validation_split=0.1,
                            callbacks=self.Ttensorboard, verbose=1)

        else:
            state_batch = np.array([_[0] for _ in batch])
            action_batch = np.array([_[1] for _ in batch])
            reward_batch = np.array([_[2] for _ in batch])
            next_state_batch = np.array([_[3] for _ in batch])
            done_batch = np.array([_[4] for _ in batch])

            # only private obs and action [np.hstack((state_batch[_,i],action_batch[_,i]))) for _ in range(batch.shape[0])]
            # all actions observable [np.hstack((state_batch[_,i],np.hstack(action_batch[_,...]))) for _ in range(batch.shape[0])]
            # everything observable [np.hstack((np.hstack(state_batch[_,...]),np.hstack(action_batch[_,...]))) for _ in range(batch.shape[0])]

            # and do the model fit
            for i in range(self.n):
                print("Training agent " + str(i))
                self.tmodel[i].fit(np.array([np.hstack((state_batch[_,i],np.hstack(action_batch[_,...])))
                                             for _ in range(batch.shape[0])]),
                                   np.array([np.hstack(next_state_batch[_, i]) for _ in range(batch.shape[0])]),
                                batch_size=minibatch_size,
                                epochs=self.net_train_epochs,
                                validation_split=0.1,
                                callbacks=self.Ttensorboard, verbose=1)

        # TODO Currently predicts reward based on state input data.
        #  Should we consider making reward predictions action-dependent too?
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
            self.memory.append((state, action, reward, next_state, done * 1))

            if done:  # and np.random.rand() <= .5:  # TODO super hacky way to get 0 rewards in cartpole
                state = environment.reset()
            else:
                state = next_state

    def setup_batch_for_RNN(self, batch):
        batch_size = batch.shape[0]
        array_size = batch_size - self.sequence_length
        x_seq = np.empty((array_size, self.sequence_length, 2*self.n))
        y_seq = np.empty((array_size, self.n))
        actual_size = 0
        state_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        next_state_batch = np.array([_[3] for _ in batch])
        done_batch = np.array([_[4] for _ in batch])
        for i in range(array_size):
            if not done_batch[i:i + self.sequence_length - 1].any():
                s_and_a = np.hstack((state_batch[i:i + self.sequence_length,...], action_batch[i:i + self.sequence_length, ...]))
                x_seq[actual_size, ...] = s_and_a[np.newaxis,...]
                y_seq[actual_size, ...] = np.expand_dims(next_state_batch[i + self.sequence_length,...], axis=0)
                actual_size += 1
        x_seq = np.resize(x_seq, (actual_size, self.sequence_length, 2*self.n))
        y_seq = np.resize(y_seq, (actual_size, self.n))
        return x_seq, y_seq


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
        env_name = 'simple_spread'
        env = make_env.make_env(env_name)
        # print(env.action_space)
        # x = env.step([np.array([0,0,1,0,0]),np.array([0,0,1,0,0]),np.array([1,0,0,0,0])])
        # print(x)

        canary = ModelLearner(env, sequence_length=1)
        canary.run(env, rounds=8)

        #print('MSE: {}'.format(canary.evaluate(env)))
