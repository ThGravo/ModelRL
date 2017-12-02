from __future__ import division
import argparse
from PIL import Image
import numpy as np
import gym
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Input, LSTM, concatenate, Reshape, Lambda
import keras.backend as K
from rl.core import Processor

# input image dimensions
img_rows, img_cols = 105, 80
INPUT_SHAPE = (img_rows, img_cols)
WINDOW_LENGTH = 1


# IMAGE PRE-PROCESSING

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


# PARSING ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


class ModelLearner():
    def __init__(self, state_shape, action_shape, action_num, learning_rate=.001):
        # get size of state and action
        self.state_size = state_shape
        self.action_shape = action_shape
        self.action_num = action_num

        # These are hyper parameters
        self.learning_rate = learning_rate
        self.batch_size = 10
        self.net_train_epochs = 64

        # create main model and target model
        self.model = self.build_model(state_shape, action_shape)

    def build_mt_model(self, state_shape, action_shape):
        image_in = Input(shape=state_shape, name='image_input')
        action_in = Input(shape=action_shape, name='action_input')

        # Convolutional layers
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(image_in)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
        conv_out = Flatten()(conv3)
        dense_out = Dense(512, activation='relu')(conv_out)
        dense_out = Reshape((1, 512))(dense_out)
        action_in_reshape = Reshape((1, 1))(action_in)
        dense_out_and_action = concatenate([dense_out, action_in_reshape], name='encoded_state_and_action')

        ## Recurrent encoding
        lstm_out = LSTM(512)(dense_out_and_action)

        ## Next-state predictor
        state_pred = Dense(512, activation='relu', name='predicted_next_state')(lstm_out)

        # Q-values predictor
        q_out = Dense(self.action_num, activation='linear')(lstm_out)

        # Specify input and outputs for model
        mt_model = Model(inputs=[image_in, action_in], outputs=[state_pred, q_out])
        print(mt_model.summary())
        mt_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return mt_model

    def build_model(self, state_shape, action_shape):
        image_in = Input(shape=state_shape, name='image_input')
        action_in = Input(shape=action_shape, name='action_input')

        # Convolutional layers
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(image_in)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
        conv_out = Flatten()(conv3)
        dense_out = Dense(512, activation='relu')(conv_out)
        dense_out = Reshape((1, 512))(dense_out)
        action_in_reshape = Reshape((1, 1))(action_in)
        dense_out_and_action = concatenate([dense_out, action_in_reshape], name='encoded_state_and_action')

        ## Recurrent encoding
        lstm_out = LSTM(512)(dense_out_and_action)

        ## Next-state predictor
        state_pred = Dense(512, activation='relu', name='predicted_next_state')(lstm_out)

        # Q-values predictor
        q_out = Dense(self.action_num, activation='linear')(dense_out)

        # Specify input and outputs for model
        model = Model(inputs=[image_in, action_in], outputs=[state_pred])
                    #outputs=[state_pred, q_out]) # TODO: Bring back Q-learning
        print(model.summary())
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return model


def get_rollouts(env, batch_size, seed=None):
    rollout_limit = env.spec.timestep_limit
    states, actions, rewards = [], [], []
    for j in range(batch_size):
        env.seed(seed)
        s = env.reset()
        rollout_s, rollout_a, rollout_r = [], [], []
        for i in range(rollout_limit):
            a = env.action_space.sample()
            s1, r, done, _ = env.step(a)
            rollout_s.append(s)
            rollout_a.append(a)
            rollout_r.append(r)
            s = s1
            if done: break
        env.seed(None)
        states.append(np.array(rollout_s))
        actions.append(np.array(rollout_a))
        rewards.append(np.array(rollout_r))
    return np.array(states), np.array(actions), np.array(rewards)


# training settings

epochs = 1000  # number of training batches
batch_size = 10  # number of rollouts per training batch
rollout_limit = env.spec.timestep_limit  # max rollout length
discount_factor = 1.00  # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.001  # you know this by now
early_stop_loss = 0  # stop training if loss < early_stop_loss, 0 or False to disable

# train policy network

if __name__ == "__main__":
    for env_name in ['Breakout-v0']:
        env = gym.make(env_name)
        # get size of state and action from environment
        state_shape = env.observation_space.shape
        action_shape = (1,)  # TODO: get shape from environment. something like env.action.space.shape?
        num_discrete_actions = env.action_space.n
        agent = ModelLearner(state_shape, action_shape, num_discrete_actions)  # TODO: prepro of frames
        states, actions, rewards = [], [], []  # lists of rollouts
        for epoch in range(epochs):
            # fill batch with rollouts
            states, actions, rewards = get_rollouts(env, batch_size, seed=epoch)
            # iterate over rollouts
            for rollout_num in range(batch_size):
            # iterate over states in rollout
                for i in range(len(states[rollout_num])-1):
                    state1 = np.reshape(states[rollout_num][i], (1,) + states[rollout_num][i].shape)
                    state2 = np.reshape(states[rollout_num][i+1], (1,) + states[rollout_num][i+1].shape)
                    states_tr = np.concatenate((state1,state2),axis=0)
                    action = np.reshape(actions[rollout_num][i], (1,) + actions[rollout_num][i].shape)
                    # get label for state prediction
                    target = agent.model.predict([state1, action], batch_size=1)
                    agent.model.fit([states_tr,action],
                              [target],
                              batch_size=1,
                              #validation_split=0.1,
                              verbose=1)
            '''
            get_layer_output = K.function([agent.model.layers[0].input, agent.model.layers[8].input],
                                          [agent.model.layers[11].output])
            layer_output = get_layer_output([states, actions])

            # Testing
            test = np.random.random(input_shape)[np.newaxis, ...]
            layer_outs = functor([test, 1.])

            target_states = intermediate_layer_model.predict([states, actions])
            # do the model fit
            agent.fit([actions, states],

                      batch[:, -model.state_size - 1:-1],
                      batch_size=batch_size,
                      validation_split=0.1,
                      verbose=0)
            '''

            #print('MSE: {}'.format(agent.model.evaluate(env)))
