import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import TensorBoard

EPISODES = 30


# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64 * 3
        self.net_train_epochs = 3
        self.train_start = 100
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.qmodel = self.build_qmodel()
        self.target_model = self.build_qmodel()
        self.tmodel = self.build_tmodel()
        self.rmodel = self.build_rmodel()
        self.dmodel = self.build_dmodel()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.qmodel.load_weights("./save_model/cartpole_dqn.h5")

        self.tensorboard = TensorBoard(log_dir='./logs',
                                       histogram_freq=1,
                                       write_graph=True,
                                       write_images=False)

    # approximate Q function
    # state is input and Q Value of each action is output
    def build_qmodel(self):
        qmodel = Sequential()
        qmodel.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        qmodel.add(Dense(24, activation='relu',
                         kernel_initializer='he_uniform'))
        qmodel.add(Dense(self.action_size, activation='linear',
                         kernel_initializer='he_uniform'))
        qmodel.summary()
        qmodel.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return qmodel

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

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.qmodel.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.qmodel.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
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

        target = self.qmodel.predict(update_input[:, :-1])
        target_val = self.target_model.predict(update_target)

        for i in range(batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        val_acc = []

        self.qmodel.fit(update_input[:, :-1], target, batch_size=batch_size,
                        epochs=self.net_train_epochs,
                        verbose=0,
                        validation_split=0.1,  # callbacks=[self.tensorboard]
                        )

        self.tmodel.fit(update_input, update_target, batch_size=batch_size,
                        epochs=self.net_train_epochs,
                        verbose=0,
                        validation_split=0.1, callbacks=[self.tensorboard]
                        )

        train_history = self.rmodel.fit(update_input[:, :-1], reward, batch_size=batch_size,
                                        epochs=self.net_train_epochs,
                                        verbose=0,
                                        validation_split=0.1,# callbacks=[self.tensorboard]
                                        )
        # TODO Currently predicts reward based on state input data. Should consider making reward predictions action-dependent too.
        # TODO Input dmodel into rmodel
        self.dmodel.fit(update_input[:, :-1], done, batch_size=batch_size,
                        epochs=self.net_train_epochs,
                        verbose=0,
                        validation_split=0.1,  # callbacks=[self.tensorboard]
                        )

        val_acc.append(train_history.history['val_acc'])
        return val_acc


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores, episodes, val_accs, episodes_val = [], [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            val_acc = agent.train_model()

            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                '''
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
                '''
                if val_acc is not None:
                    val_accs.append(val_acc[-1])
                    episodes_val.append(e)
                    pylab.plot(episodes_val, val_accs, 'b')
                    pylab.savefig("./save_graph/cartpole_dqn_val_accs.png")
                    if val_acc is not None:
                        print("episode:", e, "  val_acc:", val_acc[-1])

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            agent.qmodel.save_weights("./save_model/cartpole_dqn.h5")
