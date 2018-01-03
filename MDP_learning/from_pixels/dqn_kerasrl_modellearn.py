from __future__ import division
import argparse
import random
import time
import numpy as np
import gym

from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Permute, Input, LSTM, concatenate, Reshape
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from MDP_learning.helpers.custom_metrics import COD, NRMSE, Rsquared
from MDP_learning.from_pixels.atari_preprocessor import AtariProcessor
from MDP_learning.from_pixels.synth_env import SynthEnv

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
nb_steps_dqn_fit = 1834567  # 1750000
nb_steps_warmup_dqn_agent = int(max(0, np.sqrt(nb_steps_dqn_fit))) * 42 + 42  # 50000
target_model_update_dqn_agent = int(max(0, np.sqrt(nb_steps_dqn_fit))) * 8 + 8  # 10000
memory_limit = nb_steps_dqn_fit  # 1000000
nb_steps_annealed_policy = int(nb_steps_dqn_fit / 2)  # 1000000
ml_model_epochs = 30

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
parser = argparse.ArgumentParser()

parser.add_argument('--mode', choices=['train', 'test'], default='test')
parser.add_argument('--env-name', type=str, default='SeaquestDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()
print(args)

env_name = args.env_name
weights_filename = 'dqn_{}_weights.h5f'.format(env_name)
checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
log_filename = 'dqn_{}_log.json'.format(env_name)


def setupDQN(nb_actions):
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

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=memory_limit, window_length=WINDOW_LENGTH)

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

    return dqn, hstate_size


def trainDQN(dqn):
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=nb_steps_dqn_fit, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    # dqn.test(env, nb_episodes=1, visualize=False)


def trainML(dqn, sequence_length, hstate_size, layer_width=1024):
    # Model learner network
    action_shape = (sequence_length, 1)  # TODO: get shape from environment. something like env.action.space.shape?
    action_in = Input(shape=action_shape, name='action_input')
    enc_state = Input(shape=(sequence_length, hstate_size), name='enc_state')
    if sequence_length > 1:
        action_in_reshape = Reshape((sequence_length, 1))(action_in)
        enc_state_and_action = concatenate([enc_state, action_in_reshape], name='encoded_state_and_action')
        lstm_out = LSTM(layer_width, activation='relu')(enc_state_and_action)
    else:
        action_in_flat = Flatten(name='flat_act')(action_in)
        enc_state_flat = Flatten(name='flat_state')(enc_state)
        enc_state_and_action = concatenate([enc_state_flat, action_in_flat], name='encoded_state_and_action')
        dense_out = Dense(layer_width, activation='relu')(enc_state_and_action)
        dense_out = Dense(layer_width, activation='relu')(dense_out)
        lstm_out = Dense(int(layer_width / 2), activation='relu')(dense_out)

    state_pred = Dense(hstate_size, activation='linear', name='predicted_next_state')(lstm_out)
    reward_pred = Dense(1, activation='linear', name='predicted_reward')(lstm_out)
    terminal_pred = Dense(1, activation='sigmoid', name='predicted_terminal')(lstm_out)
    ml_model = Model(inputs=[enc_state, action_in], outputs=[state_pred, reward_pred, terminal_pred])
    ml_model.compile('adam', loss={'predicted_next_state': 'mse',
                                   'predicted_reward': 'mse',
                                   'predicted_terminal': 'binary_crossentropy'},
                     metrics=['mse', 'mae', COD, NRMSE, Rsquared])
    print(ml_model.summary())

    log_string = '_{}_slen{}_lwidth{}-{}'.format(env_name, sequence_length, layer_width, time.time())

    ########################################################################################################################
    model_truncated = Model(inputs=dqn.model.input, outputs=dqn.model.get_layer('flat_feat').output)
    print(model_truncated.summary())

    data_size = dqn.memory.observations.length
    batch_size = int(data_size / 5)
    n_rounds = 2 * int(data_size / batch_size) + 1  # go through data 2 times
    for ii in range(n_rounds):
        hstates = np.empty((batch_size, sequence_length, hstate_size), dtype=np.float32)
        actions = np.empty((batch_size, sequence_length, 1), dtype=np.float32)
        next_hstate = np.empty((batch_size, hstate_size), dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        terminals = np.empty((batch_size, 1), dtype=np.float32)

        for jj in range(batch_size):
            # check for terminals
            start = random.randrange(dqn.memory.window_length + 1, data_size - sequence_length)
            batch_idxs = range(start, start + sequence_length)
            if np.min(batch_idxs) >= dqn.memory.window_length + 1:
                experiences = dqn.memory.sample(sequence_length, batch_idxs)
            else:
                assert False
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

        with open("{}DataStats.txt".format(env_name), "w") as text_file:
            print("Var: {}".format(np.var(next_hstate)), file=text_file)
            print("Min: {}".format(np.mean(np.min(next_hstate, axis=0))), file=text_file)
            print("Max: {}".format(np.mean(np.max(next_hstate, axis=0))), file=text_file)
            print("Min: {}".format(np.min(next_hstate, axis=0)), file=text_file)
            print("Max: {}".format(np.max(next_hstate, axis=0)), file=text_file)

        ml_model.fit([hstates, actions], [next_hstate, rewards, terminals], validation_split=0.1, verbose=1,
                     epochs=ml_model_epochs, callbacks=[TensorBoard(log_dir='./logs/Tlearn'.format(log_string))])
    return ml_model, model_truncated
    # #######################################################################################################################


def dyna_train(nb_actions, ml_model, model_truncated, sequence_length, hstate_size):
    env2 = SynthEnv(ml_model, model_truncated, env, processor, sequence_length, WINDOW_LENGTH)

    hidden_in = Input(shape=(1, hstate_size), name='hidden_input')
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
    '''dyna_weights_filename = 'dyna_dqn_{}_weights.h5f'.format(env_name)
    dyna_checkpoint_weights_filename = 'dyna_dqn_' + env_name + '_weights_{step}.h5f'
    dyna_log_filename = 'dyna_dqn_{}_log.json'.format(env_name)
    dyna_callbacks = [ModelIntervalCheckpoint(dyna_checkpoint_weights_filename, interval=250000)]
    dyna_callbacks += [FileLogger(dyna_log_filename, interval=100)]'''
    dqn2.fit(env2, nb_steps=nb_steps_dqn_fit, log_interval=10000)  # callbacks=dyna_callbacks,
    return dqn2
    # #######################################################################################################################


def validate(nb_actions, model_truncated, dqn, dqn2):
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

def loadDQN(dqn):
    filename = 'dqn_{}_weights.h5f'.format(env_name)
    if args.weights:
        filename = args.weights
    dqn.load_weights(filename)


if __name__ == "__main__":

    seq_len = 10

    env = gym.make(env_name)
    np.random.seed(123)
    env.seed(123)
    num_actions = env.action_space.n

    # (gray-)scale
    processor = AtariProcessor(INPUT_SHAPE)
    dqn_agent, hidden_state_size = setupDQN(num_actions)

    if args.mode == 'train':
        trainDQN(dqn_agent)

    elif args.mode == 'test':
        loadDQN(dqn_agent)
        while dqn_agent.memory.nb_entries < memory_limit:
            dqn_agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=nb_steps_dqn_fit)

    for seq_len in [1, 3, 10, 20]:
        for width in [512, 1024, 4096]:
            dynamics_model, dqn_convolutions = trainML(dqn_agent,
                                                       sequence_length=seq_len,
                                                       hstate_size=hidden_state_size,
                                                       layer_width=width)

    # dqn_agent2 = dyna_train(num_actions, dynamics_model, dqn_convolutions, seq_len, hidden_state_size)
    # validate(num_actions, dqn_convolutions, dqn_agent, dqn_agent2)
