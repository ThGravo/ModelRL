from __future__ import division
import argparse
import random

from PIL import Image
import numpy as np
import gym

from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Permute, Input, LSTM, concatenate, Reshape
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
nb_steps_dqn_fit = 123456  # 1750000
nb_steps_warmup_dqn_agent = int(max(0, np.sqrt(nb_steps_dqn_fit))) * 42 + 42  # 50000
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

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

image_in = Input(shape=input_shape, name='main_input')
input_perm = Permute((2, 3, 1), input_shape=input_shape)(image_in)
conv1 = Conv2D(32, (8, 8), activation="relu", strides=(4, 4))(input_perm)
conv2 = Conv2D(64, (4, 4), activation="relu", strides=(2, 2))(conv1)
conv3 = Conv2D(64, (3, 3), activation="relu", strides=(1, 1))(conv2)
conv_out = Flatten(name='flat_feat')(conv3)
dense_out = Dense(512, activation='relu')(conv_out)
q_out = Dense(nb_actions, activation='linear')(dense_out)
model = Model(inputs=[image_in], outputs=[q_out])
print(model.summary())
hstate_size = int(np.prod(conv3.shape[1:]))

# Model learner network
USE_LSTM = True
sequence_length = 5 if USE_LSTM else 1
action_shape = (sequence_length, 1)  # TODO: get shape from environment. something like env.action.space.shape?
action_in = Input(shape=action_shape, name='action_input')
enc_state = Input(shape=(sequence_length, hstate_size), name='enc_state')
if USE_LSTM:
    action_in_reshape = Reshape((sequence_length, 1))(action_in)
    enc_state_and_action = concatenate([enc_state, action_in_reshape], name='encoded_state_and_action')
    lstm_out = LSTM(4096, activation='relu')(enc_state_and_action)
else:
    action_in_flat = Flatten(name='flat_act')(action_in)
    enc_state_flat = Flatten(name='flat_state')(enc_state)
    enc_state_and_action = concatenate([enc_state_flat, action_in_flat], name='encoded_state_and_action')
    lstm_out = Dense(8192, activation='relu')(enc_state_and_action)

state_pred = Dense(hstate_size, activation='linear', name='predicted_next_state')(lstm_out)
reward_pred = Dense(1, activation='linear', name='predicted_reward')(lstm_out)
terminal_pred = Dense(1, activation='sigmoid', name='predicted_terminal')(lstm_out)
ml_model = Model(inputs=[enc_state, action_in], outputs=[state_pred, reward_pred, terminal_pred])
ml_model.compile('adam', loss={'predicted_next_state': 'mse', 'predicted_reward': 'mse',
                               'predicted_terminal': 'binary_crossentropy'}, metrics=['accuracy'])
print(ml_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=memory_limit, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=nb_steps_annealed_policy)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=nb_steps_warmup_dqn_agent, gamma=.99,
               target_model_update=target_model_update_dqn_agent,
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
    dqn.fit(env, callbacks=callbacks, nb_steps=nb_steps_dqn_fit, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    # dqn.test(env, nb_episodes=1, visualize=False)
    ########################################################################################################################
    model_truncated = Model(inputs=dqn.model.input, outputs=dqn.model.get_layer('flat_feat').output)
    print(model_truncated.summary())

    data_size = dqn.memory.observations.length
    batch_size = int(data_size / 5)
    n_rounds = 2 * int(data_size / batch_size) + 1  # go through data 2 times
    for ii in range(n_rounds):
        hstates = np.empty((batch_size, sequence_length, int(np.prod(conv3.shape[1:]))), dtype=np.float32)
        actions = np.empty((batch_size, sequence_length, 1), dtype=np.float32)
        next_hstate = np.empty((batch_size, int(np.prod(conv3.shape[1:]))), dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        terminals = np.empty((batch_size, 1), dtype=np.float32)

        for jj in range(batch_size):
            # check for terminals
            start = random.randrange(data_size - sequence_length)
            experiences = dqn.memory.sample(sequence_length, range(start, start + sequence_length))
            while np.array([e.terminal1 for e in experiences]).any():
                if start is 0:
                    start = random.randrange(data_size - sequence_length)
                else:
                    start -= 1  # get a bit more terminal
                experiences = dqn.memory.sample(sequence_length, range(start, start + sequence_length))

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_seq = []
            state1_seq = []
            reward_seq = []
            action_seq = []
            terminal1_seq = []

            for e in experiences:
                state0_seq.append(e.state0)
                state1_seq.append(e.state1)
                reward_seq.append(e.reward)
                action_seq.append(e.action)
                terminal1_seq.append(e.terminal1)

            # TODO consider differencing
            state0_seq = dqn.process_state_batch(state0_seq)
            state1_seq = dqn.process_state_batch(state1_seq)
            reward_seq = np.array(reward_seq)
            action_seq = np.array(action_seq, dtype=np.float32)
            terminal1_seq = np.array(terminal1_seq)

            hidden_state0_seq = model_truncated.predict_on_batch(state0_seq)
            hidden_state1_seq = model_truncated.predict_on_batch(state1_seq)

            hstates[jj, ...] = hidden_state0_seq[np.newaxis, :, :]
            actions[jj, ...] = action_seq[np.newaxis, :, np.newaxis]
            next_hstate[jj, ...] = hidden_state1_seq[np.newaxis, -1, :]
            rewards[jj, ...] = reward_seq[np.newaxis, -1]
            terminals[jj, ...] = terminal1_seq[np.newaxis, -1]

        ml_model.fit([hstates, actions], [next_hstate, rewards, terminals], verbose=1, epochs=4,
                     callbacks=[TensorBoard(log_dir='./logs/TlearnBIG')])  # , shuffle=False)

    # #######################################################################################################################
    from collections import deque


    class SynthEnv():
        def __init__(self, tmodel, conv_model, real_env, processor, sequence_len):
            self.tmodel = tmodel
            self.conv_model = conv_model
            self.real_env = real_env
            self.processor = processor
            self.seq_len = sequence_len
            self.action_space = real_env.action_space
            self.observation_space = gym.spaces.Box(-10, 10, (int(np.prod(conv3.shape[1:])),))
            self.state_seq, self.action_seq = self.init_state()

        def init_state(self):
            state_seq = deque(maxlen=self.seq_len)
            act_seq = deque(maxlen=self.seq_len)  # TODO should be just one action
            self.real_env.reset()

            images = []
            for _ in range(self.seq_len + WINDOW_LENGTH):
                act_seq.append(self.real_env.action_space.sample())
                obs, rw, dn, info = self.real_env.step(act_seq[-1])
                obs = processor.process_observation(obs)
                images.append(obs)

            for i in range(self.seq_len):
                state_seq.append(
                    self.conv_model.predict(
                        np.expand_dims(np.array(images[i:i + WINDOW_LENGTH]), axis=0)
                    )
                )

            return state_seq, act_seq

        def step(self, action):
            # TODO append action before?
            self.action_seq.append(action)
            # reshape
            ssq = np.rollaxis(np.array(self.state_seq), 1)
            asq = np.expand_dims(np.expand_dims(np.array(self.action_seq), axis=0), axis=2)
            next_state, reward, done = self.tmodel.predict([ssq, asq])
            self.state_seq.append(next_state)
            # unwrap and add empty info
            return next_state[0], float(reward[0, 0]), bool(done[0, 0] > .5), {}

        # TODO done might never occur in unseen territory
        # TODO check timing t->t+1

        def reset(self):
            self.state_seq, self.action_seq = self.init_state()
            return self.state_seq[-1].flatten()


    env2 = SynthEnv(ml_model, model_truncated, env, processor, sequence_length)

    hidden_in = Input(shape=(1, int(np.prod(conv3.shape[1:]))), name='hidden_input')
    hidden_in_f = Flatten(name='flat_hidden')(hidden_in)
    dense_out = Dense(512, activation='relu')(hidden_in_f)
    q_out = Dense(nb_actions, activation='linear')(dense_out)
    model2 = Model(inputs=[hidden_in], outputs=[q_out])
    print(model2.summary())

    memory2 = SequentialMemory(limit=memory_limit, window_length=1)
    policy2 = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                   nb_steps=nb_steps_annealed_policy)
    dqn2 = DQNAgent(model=model2, nb_actions=nb_actions, policy=policy2, memory=memory2,
                    nb_steps_warmup=nb_steps_warmup_dqn_agent, gamma=.99,
                    target_model_update=target_model_update_dqn_agent,
                    train_interval=4, delta_clip=1.)
    dqn2.compile(Adam(lr=.00025), metrics=['mae'])
    dqn2.fit(env2, callbacks=callbacks, nb_steps=nb_steps_dqn_fit, log_interval=10000)

    # #######################################################################################################################

    image_in = Input(shape=input_shape, name='main_input')
    input_perm = Permute((2, 3, 1), input_shape=input_shape)(image_in)
    conv1 = Conv2D(32, (8, 8), activation="relu", strides=(4, 4))(input_perm)
    conv2 = Conv2D(64, (4, 4), activation="relu", strides=(2, 2))(conv1)
    conv3 = Conv2D(64, (3, 3), activation="relu", strides=(1, 1))(conv2)
    conv_out = Flatten(name='flat_feat')(conv3)
    dense_out = Dense(512, activation='relu')(conv_out)
    q_out = Dense(nb_actions, activation='linear')(dense_out)
    model3 = Model(inputs=[image_in], outputs=[q_out])

    # Combine truncated and model2 top
    wghts = [np.zeros(w.shape) for w in model3.get_weights()]
    for layer, w in enumerate(model_truncated.get_weights()):
        wghts[layer] = w
    depth_conv = len(model_truncated.get_weights())
    for layer, w in enumerate(dqn2.model.get_weights()):
        wghts[layer + depth_conv] = w
    model3.set_weights(wghts)
    print(model3.summary())

    memory3 = SequentialMemory(limit=memory_limit, window_length=WINDOW_LENGTH)
    policy3 = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                   nb_steps=nb_steps_annealed_policy)
    dqn3 = DQNAgent(model=model3, nb_actions=nb_actions, policy=policy3, memory=memory3,
                    processor=processor, nb_steps_warmup=nb_steps_warmup_dqn_agent, gamma=.99,
                    target_model_update=target_model_update_dqn_agent,
                    train_interval=4, delta_clip=1.)
    dqn3.compile(Adam(lr=.00025), metrics=['mae'])
    dqn.test(env, nb_episodes=10, visualize=False)
    dqn3.test(env, nb_episodes=10, visualize=False)
    # #######################################################################################################################

elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
