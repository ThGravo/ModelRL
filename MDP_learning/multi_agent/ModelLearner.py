import MDP_learning.multi_agent.policies as MAPolicies
from MDP_learning.helpers import build_models
from MDP_learning.helpers.logging_model_learner import LoggingModelLearner
from MDP_learning.helpers.model_evaluation import sk_eval

from collections import deque
import time
import numpy as np


class ModelLearner(LoggingModelLearner):
    def __init__(self, environment, agent_id, action_size,
                 mem_size=3000, epochs=4, learning_rate=.001,
                 sequence_length=0, write_tboard=True, envname=None, net_depth=2):
        super().__init__(environment, sequence_length,
                         write_tboard=write_tboard,
                         out_dir_add=
                         'agentID{}{}{}'.format(
                             agent_id,
                             '_scenario_{}'.format(envname) if envname is not None else '',
                             '_netDepth{}'.format(net_depth)))
        # self.net_depth = net_depth
        # self.learning_rate = learning_rate
        self.net_train_epochs = epochs
        # self.mem_size = mem_size
        self.learn_transitions = False
        self.learn_rewards = False
        self.learn_positions = True

        self.done_memory = deque(maxlen=mem_size)
        if self.learn_transitions:
            self.x_memory = deque(maxlen=mem_size)
            self.next_obs_memory = deque(maxlen=mem_size)
        if self.learn_rewards:
            self.obs_memory = deque(maxlen=mem_size)
            self.reward_memory = deque(maxlen=mem_size)
        if self.learn_positions:
            self.vel_rew_memory = deque(maxlen=mem_size)
            self.ent_pos_memory = deque(maxlen=mem_size)

        self.agent_id = agent_id
        self.policy = MAPolicies.RandomPolicy(environment, self.agent_id)

        dim_mult = tuple([int(128 / net_depth) + 1 for _ in range(net_depth)])

        if self.learn_transitions:
            self.tmodel = build_models.build_regression_model(
                input_dim=self.env.observation_space[self.agent_id].shape[0] + action_size,
                output_dim=self.env.observation_space[self.agent_id].shape[0],
                recurrent=self.useRNN,
                dim_multipliers=dim_mult,
                lr=learning_rate,
                activations=('relu', 'sigmoid'),
                opt_decay=0,
                opt_clipnorm=0,
                num_hlayers=net_depth - 1)
            self.models.append(self.tmodel)

        if self.learn_rewards:
            self.rmodel = build_models.build_regression_model(
                input_dim=self.env.observation_space[self.agent_id].shape[0],
                output_dim=1,
                recurrent=self.useRNN,
                dim_multipliers=dim_mult,
                lr=learning_rate,
                activations=('relu', 'relu'),
                opt_decay=0,
                opt_clipnorm=0,
                num_hlayers=net_depth - 1)
            self.models.append(self.rmodel)

        if self.learn_positions:
            # used to predict the locations of a landmarks based on movement and reward sequence
            self.dmodel = build_models.build_regression_model(
                input_dim=2 + 1,
                output_dim=self.env.observation_space[self.agent_id].shape[0] - 2,
                recurrent=self.useRNN,
                dim_multipliers=dim_mult,
                lr=learning_rate,
                activations=('relu', 'relu'),
                opt_decay=0.01,
                opt_clipnorm=1.0,
                num_hlayers=net_depth - 1)
            self.models.append(self.dmodel)

        self.save_model_config()

    def get_action(self, obs_n):
        return self.policy.action(obs_n[self.agent_id])

    def append_to_mem(self, obs, act, reward, obs_next, done):
        # TODO is there any point in learning done in this environment
        self.done_memory.append([done])

        if self.learn_transitions:
            self.x_memory.append(np.concatenate((obs, act)))
            self.next_obs_memory.append(obs_next)

        if self.learn_rewards:
            self.obs_memory.append(obs)
            # self.reward_memory.append([-np.sqrt(-reward)])
            self.reward_memory.append([reward])

        if self.learn_positions:
            self.vel_rew_memory.append(np.hstack((obs[:2], reward)))
            self.ent_pos_memory.append(obs[2:])

    def clear_mem(self):
        # TODO brittle
        self.done_memory.clear()
        if self.learn_transitions:
            self.x_memory.clear()
            self.next_obs_memory.clear()
        if self.learn_rewards:
            self.obs_memory.clear()
            self.reward_memory.clear()
        if self.learn_positions:
            self.vel_rew_memory.clear()
            self.ent_pos_memory.clear()

    def setup_batch_for_RNN(self, input_batch, signal, done):
        array_size = input_batch.shape[0] - self.sequence_length
        actual_size = 0

        d = np.array(done)
        seq = np.zeros((array_size, self.sequence_length, input_batch.shape[1]), dtype=np.float32)
        output = np.zeros((array_size, signal.shape[1]), dtype=np.float32)

        for jj in range(1, array_size, max(1, self.sequence_length)):
            if jj % 10000 == 1:
                print('Filling data for RNN: {} of {} ({} w/o terminals)'.format(jj, array_size, actual_size))
            # NO intermediate terminals
            if not d[jj:jj + self.sequence_length - 1, :].any():
                seq[actual_size, :, :] = input_batch[np.newaxis, jj:jj + self.sequence_length, :]
                output[actual_size, :] = signal[jj + self.sequence_length, :]
                actual_size += 1

        seq.resize((actual_size, self.sequence_length, input_batch.shape[1]))
        output.resize((actual_size, signal.shape[1]))
        print('Done filling the data!')
        return seq, output

    # if this becomes a bottle-neck again this could be transformed into a parallel version
    # from joblib import Parallel, delayed
    # from joblib.pool import has_shareable_memory
    # from numba import jit
    def setup_batch_for_RNN2(self, input_batch, signal, done):
        array_length = input_batch.shape[0] - self.sequence_length
        actual_size = 0

        d = np.array(done)
        seq = np.zeros((array_length, self.sequence_length, input_batch.shape[1]), dtype=np.float32)
        output = np.zeros((array_length, signal.shape[1]), dtype=np.float32)
        mask = np.zeros(array_length, dtype=np.bool_)

        for jj in range(1, array_length, max(1, self.sequence_length)):
            if jj % 10000 == 1:
                print('Filling data for RNN: {} of {} ({} w/o terminals)'.format(jj, array_length, actual_size))
            # NO intermediate terminals
            if not d[jj:jj + self.sequence_length - 1, :].any():
                mask[jj] = True
                seq[jj, :, :] = input_batch[np.newaxis, jj:jj + self.sequence_length, :]
                output[jj, :] = signal[jj + self.sequence_length, :]
                actual_size += 1

        seq_out = seq[mask]
        output_out = output[mask]
        print('Done filling the data!')
        return seq_out, output_out

    def train_models(self, minibatch_size=32):
        if self.learn_transitions:  # predictiong state transitions
            if self.useRNN:
                now = time.time()
                input_data, train_signal = self.setup_batch_for_RNN(np.array(self.x_memory),
                                                                    np.array(self.next_obs_memory),
                                                                    done=self.done_memory)
                print("setup_batch_for_RNN took: {}".format(time.time() - now))
            else:
                input_data = np.array(self.x_memory)
                train_signal = np.array(self.next_obs_memory)

            history = self.tmodel.fit(input_data,
                                      train_signal,
                                      batch_size=minibatch_size,
                                      epochs=self.net_train_epochs,
                                      validation_split=0.1,
                                      callbacks=self.Ttensorboard,
                                      verbose=1)
            sk_eval(self.tmodel, input_data, train_signal, '{}/tmodel_R2.txt'.format(self.out_dir))

        if self.learn_rewards:  # predicting rewards from observations
            if self.useRNN:
                input_data, train_signal = self.setup_batch_for_RNN(np.array(self.obs_memory),
                                                                    np.array(self.reward_memory),
                                                                    done=self.done_memory)
            else:
                input_data = np.array(self.obs_memory)
                train_signal = np.array(self.reward_memory)

            history = self.rmodel.fit(input_data,
                                      train_signal,
                                      batch_size=minibatch_size,
                                      epochs=self.net_train_epochs,
                                      validation_split=0.1,
                                      callbacks=self.Rtensorboard,
                                      verbose=1)
            sk_eval(self.rmodel, input_data, train_signal, '{}/rmodel_R2.txt'.format(self.out_dir))

            # DEBUG
            if False:
                import matplotlib
                matplotlib.use('GTK3Cairo', warn=False, force=True)
                from mpl_toolkits.mplot3d import Axes3D
                import matplotlib.pyplot as plt

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(np.array(self.obs_memory)[:, 2],
                           np.array(self.obs_memory)[:, 3],
                           np.array(self.reward_memory),
                           c='b', marker='^')

                ax.scatter(np.array(self.obs_memory)[:, 2],
                           np.array(self.obs_memory)[:, 3],
                           self.rmodel.predict(np.array(self.obs_memory)),
                           c='r', marker='o')
                plt.show()

        if self.learn_positions:  # Predicting relative position of entities from movement and rewards
            if self.useRNN:
                input_data, train_signal = self.setup_batch_for_RNN(np.array(self.vel_rew_memory),
                                                                    np.array(self.ent_pos_memory),
                                                                    done=self.done_memory)
            else:
                input_data = np.array(self.vel_rew_memory)
                train_signal = np.array(self.ent_pos_memory)

            history = self.dmodel.fit(input_data,
                                      train_signal,
                                      batch_size=minibatch_size,
                                      epochs=self.net_train_epochs,
                                      validation_split=0.1,
                                      callbacks=self.Dtensorboard,
                                      verbose=1)

            sk_eval(self.dmodel, input_data, train_signal, '{}/dmodel_R2.txt'.format(self.out_dir))

        self.save()
