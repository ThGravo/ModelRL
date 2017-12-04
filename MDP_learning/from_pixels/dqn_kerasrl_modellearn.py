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

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

image_in = Input(shape=input_shape, name='main_input')
input_perm = Permute((2, 3, 1), input_shape=input_shape)(image_in)
conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(input_perm)
conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(conv1)
conv3 = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')(conv2)
conv_out = Flatten(name='flat_feat')(conv3)
dense_out = Dense(512, activation='relu')(conv_out)
q_out = Dense(nb_actions, activation='linear')(dense_out)
model = Model(inputs=[image_in], outputs=[q_out])
print(model.summary())

# Model learner network
sequence_length = 10
action_shape = (sequence_length, 1)  # TODO: get shape from environment. something like env.action.space.shape?
action_in = Input(shape=action_shape, name='action_input')
enc_state = Input(shape=(sequence_length, int(np.prod(conv3.shape[1:]))), name='enc_state')
action_in_reshape = Reshape((sequence_length, 1))(action_in)
enc_state_and_action = concatenate([enc_state, action_in_reshape], name='encoded_state_and_action')
lstm_out = LSTM(512)(enc_state_and_action)
state_pred = Dense(int(np.prod(conv3.shape[1:])), activation='relu', name='predicted_next_state')(lstm_out)
ml_model = Model(inputs=[enc_state, action_in], outputs=[state_pred])
ml_model.compile('rmsprop', 'mae', metrics=['accuracy'])
print(ml_model.summary())

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
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=100000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    # dqn.test(env, nb_episodes=1, visualize=False)

    model_truncated = Model(inputs=dqn.model.input, outputs=dqn.model.get_layer('flat_feat').output)
    print(model_truncated.summary())

    data_size = dqn.memory.observations.length
    batch_size = 10000
    n_epochs = 3 * round(data_size / batch_size)  # go through data 3 times
    for ii in range(n_epochs):
        hstates = np.empty((batch_size, sequence_length, int(np.prod(conv3.shape[1:]))), dtype=np.float32)
        actions = np.empty((batch_size, sequence_length, 1), dtype=np.float32)
        next_hstate = np.empty((batch_size, int(np.prod(conv3.shape[1:]))), dtype=np.float32)

        # starts = [random.randrange(data_size - (sequence_length + 1)) for i in range(batch_size)]
        # idxs = [i + j for i in starts for j in range(sequence_length + 1)]
        # n_samples = batch_size * (sequence_length + 1)
        # experiences = dqn.memory.sample(n_samples, idxs)

        curr_batch = 0
        # for jj in range(0, n_samples, sequence_length + 1):
        for jj in range(batch_size):
            # check for terminals
            start = random.randrange(data_size - (sequence_length + 1))
            experiences = dqn.memory.sample(sequence_length + 1, range(start, start + sequence_length + 1))
            while np.array([e.terminal1 for e in experiences]).any():
                start = random.randrange(data_size - (sequence_length + 1))
                experiences = dqn.memory.sample(sequence_length + 1, range(start, start + sequence_length + 1))

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_seq = []
            # state1_batch = []
            # reward_batch = []
            action_batch = []
            # terminal1_batch = []
            # for e in experiences[jj:jj + sequence_length + 1]:
            for e in experiences:
                state0_seq.append(e.state0)
                # state1_batch.append(e.state1)
                # reward_batch.append(e.reward)
                action_batch.append(e.action)
                # terminal1_batch.append(e.terminal1)

            state0_seq = dqn.process_state_batch(state0_seq)
            # state1_batch = dqn.process_state_batch(state1_batch)
            # reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch, dtype=np.float32)
            # terminal1_batch = np.array(terminal1_batch)

            hidden_states0 = model_truncated.predict_on_batch(state0_seq)

            hstates[curr_batch, ...] = hidden_states0[np.newaxis, :-1, :]
            actions[curr_batch, ...] = np.expand_dims(np.expand_dims(action_batch[:-1], axis=0), axis=2)
            next_hstate[curr_batch, ...] = hidden_states0[np.newaxis, -1, :]
            curr_batch += 1

        ml_model.fit([hstates, actions], next_hstate, verbose=1,  # epochs=8,
                     callbacks=[TensorBoard(log_dir='./logs/Tlearn')])


elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
