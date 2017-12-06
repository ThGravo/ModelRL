from __future__ import division
import argparse
import random

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input, Embedding, LSTM, concatenate, \
    Reshape
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from keras.callbacks import TensorBoard

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
nb_steps_dqn_fit = 1750000  # 1750000
nb_steps_warmup_dqn_agent = int(max(0, np.sqrt(nb_steps_dqn_fit))) * 42 + 1000  # 50000
target_model_update_dqn_agent = int(max(0, np.sqrt(nb_steps_dqn_fit))) * 8 + 8  # 10000
memory_limit = nb_steps_dqn_fit  # 1000000
nb_steps_annealed_policy = int(nb_steps_dqn_fit / 2)  # 1000000


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

class ModelBasedLearner:

    def __init__(self, env, input_shape, action_shape=(1, ), sequence_length=10):
        self.env = env
        self.sequence_length = sequence_length
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.nb_actions = self.env.action_space.n

    def main_net(self):
        self.input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
        image_in = Input(shape=self.input_shape, name='main_input')
        action_in = Input(shape=self.action_shape, name='action_input')
        input_perm = Permute((2, 3, 1), input_shape=self.input_shape)(image_in)
        conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(input_perm)
        conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(conv1)
        conv3 = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')(conv2)
        conv_out = Flatten(name='flat_feat')(conv3)
        state_and_action = concatenate([conv_out, action_in], name='state_and_action')
        dense = Dense(512, activation='relu')(state_and_action)

        # feed-forward model learning branch
        state_pred = Dense(int(np.prod(conv_out.shape[1])), activation='relu', name='predicted_next_state')(dense)
        reward_pred = Dense(1, activation='linear', name='predicted_reward')(dense)
        terminal_pred = Dense(1, activation='sigmoid', name='predicted_terminal')(dense)

        # Q-learning branch
        q_out = Dense(self.nb_actions, activation='linear')(dense)

        main_model = Model(inputs=[image_in, action_in], outputs=[state_pred,reward_pred,terminal_pred,q_out])
        return main_model



# Define memory and image pre-processing
memory = SequentialMemory(limit=memory_limit,window_length=WINDOW_LENGTH)
processor = AtariProcessor()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=nb_steps_annealed_policy)

main_net =

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=nb_steps_warmup_dqn_agent, gamma=.99,
               target_model_update=target_model_update_dqn_agent,
               train_interval=4, delta_clip=1.)