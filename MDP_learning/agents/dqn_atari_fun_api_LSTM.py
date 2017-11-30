from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Input, LSTM, concatenate, Reshape
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

# input image dimensions
img_rows, img_cols = 84, 84
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




# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (1, img_rows, img_cols)

image_in = Input(shape=input_shape, name='main_input')
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    input_perm = Permute((2, 3, 1), input_shape=input_shape)(image_in)
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    input_perm = Permute((1, 2, 3), input_shape=input_shape)(image_in)
else:
    raise RuntimeError('Unknown image_dim_ordering.')
'''
if K.image_data_format() == 'channels_first':
    image_in = image_in.reshape(image_in.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = image_in.reshape(image_in.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
'''
conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_perm)
conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
conv_out = Flatten()(conv3)
dense_out = Dense(512, activation='relu')(conv_out)

## WITH LSTM
dense_out = Reshape((1, 512))(dense_out)
lstm_out = LSTM(512)(dense_out)
# Q-values output
q_out = Dense(nb_actions, activation='linear')(lstm_out)

### WITHOUT STATE PREDICTION
model = Model(inputs=[image_in], outputs=[q_out])

#### WITH STATE PREDICTION
# State predictor output
#action_aux = Input(shape=(1,), name='aux_input')
#state_pred_in = concatenate([lstm_out, action_aux])
#state_out = Dense(input_shape, activation='linear')(lstm_out)
#model = Model(inputs=[image_in], outputs=[q_out, state_out])

print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)#
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
