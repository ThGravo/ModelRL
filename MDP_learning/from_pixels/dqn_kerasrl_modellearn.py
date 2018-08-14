from __future__ import division
import argparse
import random
import time
import os
import numpy as np
import gym

from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Permute, Input, LSTM, concatenate, Reshape
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from MDP_learning.from_pixels.atari_config import AtariConfig
from MDP_learning.from_pixels.dqn_agent import setupDQN, trainDQN
from MDP_learning.from_pixels.trainICM import train_icm
from MDP_learning.helpers.custom_metrics import COD, NRMSE, Rsquared
from MDP_learning.from_pixels.atari_preprocessor import AtariProcessor
from MDP_learning.from_pixels.synth_env import SynthEnv


def trainML(cfg, dqn, sequence_length, layer_width_multi=1, do_diff=False):
    conv_features = dqn.model.get_layer('flat_feat')
    hstate_size = np.prod(conv_features.output_shape[1:])
    layer_width = sequence_length * hstate_size * layer_width_multi
    # Model learner network
    # TODO: get shape from environment. something like env.action.space.shape?
    action_in = Input(shape=(sequence_length, 1), name='action_input')
    enc_state = Input(shape=(sequence_length, hstate_size), name='enc_state')
    if 0:#sequence_length > 1:
        action_in_reshape = Reshape((sequence_length, 1))(action_in)
        enc_state_reshape = Reshape((sequence_length, hstate_size))(enc_state)
        enc_state_and_action = concatenate([enc_state_reshape, action_in_reshape], name='encoded_state_and_action')
        lstm_out = LSTM(layer_width, activation='relu')(enc_state_and_action)
        state_pred = Dense(hstate_size, activation='linear', name='predicted_next_state')(lstm_out)
        reward_pred = Dense(1, activation='linear', name='predicted_reward')(lstm_out)
        terminal_pred = Dense(1, activation='sigmoid', name='predicted_terminal')(lstm_out)
    else:
        action_in_flat = Flatten(name='flat_act')(action_in)
        enc_state_flat = Flatten(name='flat_state')(enc_state)
        enc_state_and_action = concatenate([enc_state_flat, action_in_flat], name='encoded_state_and_action')
        dense_out = Dense(layer_width, activation='relu')(enc_state_and_action)
        dense_out = Dense(int(layer_width / 4), activation='relu')(dense_out)
        dense_out = Dense(int(layer_width / 8), activation='relu')(dense_out)
        state_pred = Dense(hstate_size, activation='linear', name='predicted_next_state')(dense_out)
        #state_pred = Reshape((1, hstate_size), name='predicted_next_state')(state_pred)
        reward_pred = Dense(1, activation='linear', name='predicted_reward')(dense_out)
        #reward_pred = Reshape((1, 1), name='predicted_reward')(reward_pred)
        terminal_pred = Dense(1, activation='sigmoid', name='predicted_terminal')(dense_out)
        #terminal_pred = Reshape((1, 1), name='predicted_terminal')(terminal_pred)

    ml_model = Model(inputs=[enc_state, action_in], outputs=[state_pred, reward_pred, terminal_pred])
    ml_model.compile('adam', loss={'predicted_next_state': 'mse',
                                   'predicted_reward': 'mse',
                                   'predicted_terminal': 'binary_crossentropy'},
                     metrics=['mse', 'mae', COD, NRMSE, Rsquared])
    print(ml_model.summary())

    log_string = 'flat_feat_{}_slen{}_lwidth{}-{}'.format(cfg.env_name, sequence_length, layer_width, time.time())
    print('logging to {}'.format(log_string))
    ########################################################################################################################
    model_truncated = Model(inputs=dqn.model.input, outputs=conv_features.output)
    print(model_truncated.summary())

    data_size = dqn.memory.observations.length
    chunk_size = int(data_size / 100) + 1
    n_rounds = 2 * int(data_size / chunk_size) + 1  # go through data 2 times
    if do_diff:
        sample_length = sequence_length + 1
    else:
        sample_length = sequence_length
    for ii in range(n_rounds):
        print("{} of {} n_rounds".format(ii, n_rounds))
        hstates = np.empty((chunk_size, sequence_length, hstate_size), dtype=np.float32)
        actions = np.empty((chunk_size, sequence_length, 1), dtype=np.float32)
        next_hstate = np.empty((chunk_size, hstate_size), dtype=np.float32)
        rewards = np.empty((chunk_size, 1), dtype=np.float32)
        terminals = np.empty((chunk_size, 1), dtype=np.float32)

        for jj in range(chunk_size):
            print("{} of {} chunk_size".format(jj, chunk_size))
            # check for terminals
            start = random.randrange(dqn.memory.window_length + 1, data_size - sample_length)
            batch_idxs = range(start, start + sample_length)
            if np.min(batch_idxs) >= dqn.memory.window_length + 1:
                experiences = dqn.memory.sample(sample_length, batch_idxs)
            else:
                assert False
            while np.array([e.terminal1 for e in experiences]).any():
                if start is 0:
                    start = random.randrange(data_size - sample_length)
                else:
                    start -= 1  # get a bit more terminal
                experiences = dqn.memory.sample(sample_length, range(start, start + sample_length))

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
            if do_diff:
                hidden_state0_seq = np.reshape(
                    hidden_state0_seq[1:, ...] - hidden_state0_seq[:-1, ...], (sequence_length, hstate_size))
                hidden_state1_seq = np.reshape(
                    hidden_state1_seq[1:, ...] - hidden_state1_seq[:-1, ...], (sequence_length, hstate_size))
            else:
                hidden_state0_seq = np.reshape(hidden_state0_seq, (sequence_length, hstate_size))
                hidden_state1_seq = np.reshape(hidden_state1_seq, (sequence_length, hstate_size))

            hstates[jj, :, :] = hidden_state0_seq
            if do_diff:
                actions[jj, :, :] = action_seq[1:, np.newaxis]
            else:
                actions[jj, :, :] = action_seq[:, np.newaxis]
            next_hstate[jj, :] = hidden_state1_seq[-1, :]
            rewards[jj, :] = reward_seq[-1]
            terminals[jj, :] = terminal1_seq[-1]

        if False:  # Debug
            with open("{}DataStats.txt".format(cfg.env_name), "w") as text_file:
                print("Var: {}".format(np.var(next_hstate)), file=text_file)
                print("Min: {}".format(np.mean(np.min(next_hstate, axis=0))), file=text_file)
                print("Max: {}".format(np.mean(np.max(next_hstate, axis=0))), file=text_file)
                print("Min: {}".format(np.min(next_hstate, axis=0)), file=text_file)
                print("Max: {}".format(np.max(next_hstate, axis=0)), file=text_file)

        ml_model.fit([hstates, actions], [next_hstate, rewards, terminals],
                     validation_split=0.2, verbose=1, epochs=cfg.ml_model_epochs, callbacks=[
                TensorBoard(log_dir='./dqn_logs/{}/{}'.format(os.path.splitext(cfg.filename)[0], log_string))])
    return ml_model, model_truncated
    # #######################################################################################################################


def dyna_train(cfg, nb_actions, ml_model, model_truncated, sequence_length, hstate_size, processor):
    env2 = SynthEnv(ml_model, model_truncated, cfg.env, processor, sequence_length, cfg.WINDOW_LENGTH)

    hidden_in = Input(shape=(1, hstate_size), name='hidden_input')
    hidden_in_f = Flatten(name='flat_hidden')(hidden_in)
    dense_out = Dense(512, activation='relu')(hidden_in_f)
    q_out = Dense(nb_actions, activation='linear')(dense_out)
    model2 = Model(inputs=[hidden_in], outputs=[q_out])
    print(model2.summary())

    memory2 = SequentialMemory(limit=cfg.memory_limit, window_length=1)
    policy2 = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                   nb_steps=cfg.nb_steps_annealed_policy)
    dqn2 = DQNAgent(model=model2, nb_actions=nb_actions, policy=policy2, memory=memory2,
                    nb_steps_warmup=cfg.nb_steps_warmup_dqn_agent, gamma=.99,
                    target_model_update=cfg.target_model_update_dqn_agent,
                    train_interval=4, delta_clip=1.)
    dqn2.compile(Adam(lr=.00025), metrics=['mae'])
    '''dyna_weights_filename = 'dyna_dqn_{}_weights.h5f'.format(env_name)
    dyna_checkpoint_weights_filename = 'dyna_dqn_' + env_name + '_weights_{step}.h5f'
    dyna_log_filename = 'dyna_dqn_{}_log.json'.format(env_name)
    dyna_callbacks = [ModelIntervalCheckpoint(dyna_checkpoint_weights_filename, interval=250000)]
    dyna_callbacks += [FileLogger(dyna_log_filename, interval=100)]'''
    dqn2.fit(env2, nb_steps=cfg.nb_steps_dqn_fit, log_interval=10000)  # callbacks=dyna_callbacks,
    return dqn2
    # #######################################################################################################################


def validate(cfg, env, nb_actions, model_truncated, dqn, dqn2):
    image_in = Input(shape=cfg.input_shape, name='main_input')
    input_perm = Permute((2, 3, 1), input_shape=cfg.input_shape)(image_in)
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

    memory3 = SequentialMemory(limit=cfg.memory_limit, window_length=cfg.WINDOW_LENGTH)
    policy3 = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                   nb_steps=cfg.nb_steps_annealed_policy)
    dqn3 = DQNAgent(model=model3, nb_actions=nb_actions, policy=policy3, memory=memory3,
                    processor=processor, nb_steps_warmup=cfg.nb_steps_warmup_dqn_agent, gamma=.99,
                    target_model_update=cfg.target_model_update_dqn_agent,
                    train_interval=4, delta_clip=1.)
    dqn3.compile(Adam(lr=.00025), metrics=['mae'])
    dqn.test(env, nb_episodes=10, visualize=False)
    dqn3.test(env, nb_episodes=10, visualize=False)


# #######################################################################################################################

def loadDQN(cfg, dqn):
    print('loading: {}'.format(cfg.filename))
    dqn.load_weights(cfg.filename)


if __name__ == "__main__":
    print('START')
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='PongDeterministic-v4')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--width_multi', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=2)
    args = parser.parse_args()
    print(args)

    atariCfg = AtariConfig(args.env_name, args.weights)
    environment = gym.make(atariCfg.env_name)
    print('Playing: {}'.format(environment))
    np.random.seed(123)
    environment.seed(123)
    num_actions = environment.action_space.n

    # (gray-)scale
    processor = AtariProcessor(atariCfg.INPUT_SHAPE)
    dqn_agent = setupDQN(atariCfg, num_actions, processor)

    if args.mode == 'train':
        if args.weights is not None:
            loadDQN(atariCfg, dqn_agent)
        trainDQN(atariCfg, environment, dqn_agent)

    elif args.mode == 'test':
        loadDQN(atariCfg, dqn_agent)
        while dqn_agent.memory.nb_entries < atariCfg.memory_limit:
            print("{} samples of {}".format(dqn_agent.memory.nb_entries, atariCfg.memory_limit))
            dqn_agent.test(environment, nb_episodes=1, visualize=False, nb_max_episode_steps=atariCfg.nb_steps_dqn_fit)

    print('Network width multiplier: {}'.format(args.width_multi))
    print('Sequence length: {}'.format(args.seq_len))
    # for seq_len in [1, 3, 10]:
    '''dynamics_model, dqn_convolutions = trainML(atariCfg, dqn_agent,
                                               sequence_length=args.seq_len,
                                               layer_width_multi=args.width_multi)
    '''
    im_model = train_icm(atariCfg, dqn_agent, num_actions)

    # dqn_agent2 = dyna_train(num_actions, dynamics_model, dqn_convolutions, seq_len, hidden_state_size)
    # validate(num_actions, dqn_convolutions, dqn_agent, dqn_agent2)
