from MDP_learning.multi_agent import make_env2
import MDP_learning.multi_agent.policies as MAPolicies
from MDP_learning.helpers import build_models
from MDP_learning.helpers.logging_model_learner import LoggingModelLearner

from collections import deque
import random
import numpy as np


class ModelLearner(LoggingModelLearner):
    def __init__(self, environment, agent_id, action_size,
                 mem_size=3000, epochs=4, learning_rate=.001,
                 sequence_length=0, write_tboard=True, envname=None):
        super().__init__(environment, sequence_length,
                         write_tboard=write_tboard,
                         out_dir_add=
                         'agentID{}{}'.format(
                             agent_id,
                             '_scenario_{}'.format(envname) if envname is not None else ''))

        self.learning_rate = learning_rate
        self.net_train_epochs = epochs
        self.mem_size = mem_size

        self.x_memory = deque(maxlen=mem_size)
        self.next_obs_memory = deque(maxlen=mem_size)
        self.obs_memory = deque(maxlen=mem_size)
        self.reward_memory = deque(maxlen=mem_size)
        self.done_memory = deque(maxlen=mem_size)

        self.agent_id = agent_id
        self.policy = MAPolicies.RandomPolicy(env, self.agent_id)

        self.tmodel = build_models.build_regression_model(
            input_dim=self.env.observation_space[self.agent_id].shape[0] + action_size,
            output_dim=self.env.observation_space[self.agent_id].shape[0],
            recurrent=self.useRNN)

        self.rmodel = build_models.build_regression_model(
            input_dim=self.env.observation_space[self.agent_id].shape[0],
            output_dim=1,
            recurrent=self.useRNN,
            # dim_multipliers=(320, 160),
            # activations=('relu', 'relu')
        )
        # used to predict the locations of a landmarks based on movement and reward sequence
        self.dmodel = build_models.build_regression_model(
            input_dim=3,
            output_dim=2,
            recurrent=self.useRNN)

        self.models = [self.tmodel]
        self.save_model_config()

    def get_action(self, obs_n):
        return self.policy.action(obs_n[self.agent_id])

    def append_to_mem(self, obs, act, reward, obs_next, done):
        # TODO is there any point in learning done in this environment
        self.x_memory.append(np.concatenate((obs, act)))
        self.next_obs_memory.append(obs_next)
        self.obs_memory.append(obs)
        # self.reward_memory.append([-np.sqrt(-reward)])
        self.reward_memory.append([reward])
        self.done_memory.append([done])

    def clear_mem(self):
        self.x_memory.clear()
        self.next_obs_memory.clear()

    def setup_batch_for_RNN(self, input_batch, signal, done):
        array_size = input_batch.shape[0] - self.sequence_length
        actual_size = 0

        seq = np.empty((array_size, self.sequence_length, input_batch.shape[1]))
        output = np.empty((array_size, signal.shape[1]))

        for jj in range(array_size):
            # NO intermediate terminals
            if not np.array(done)[jj:jj + self.sequence_length - 1, :].any():
                seq[actual_size, ...] = input_batch[np.newaxis, jj:jj + self.sequence_length, :]
                output[actual_size, ...] = signal[jj + self.sequence_length, :]
                actual_size += 1

        seq.resize((actual_size, self.sequence_length, input_batch.shape[1]))
        output.resize((actual_size, signal.shape[1]))
        return seq, output

    def train_models(self, minibatch_size=32):
        if False:
            if self.useRNN:
                input_data, train_signal = self.setup_batch_for_RNN(np.array(self.x_memory),
                                                                    np.array(self.next_obs_memory),
                                                                    done=self.done_memory)
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
        else:
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

        # NRMSE
        denom = np.array(self.reward_memory).max() - np.array(self.reward_memory).min()

        # COD
        # denom =

        self.save()


class MultiAgentModelLearner(LoggingModelLearner):
    def __init__(self, environment, mem_size=3000, epochs=4, learning_rate=.001,
                 sequence_length=0, write_tboard=True, scenario_name=None):
        super().__init__(environment, sequence_length,
                         write_tboard=write_tboard,
                         out_dir_add='scenario_name{}'.format(scenario_name) if scenario_name is not None else None)
        self.render = True
        self.joined_actions = False
        self.gather_joined_mem = False
        # how likely a random reset is (1 disables it, 2 is resetting always)
        self.reset_randomrange = 1  # int(mem_size / 1000) + 2
        self.mem_size = mem_size
        self.x_memory = deque(maxlen=mem_size if self.gather_joined_mem else 1)
        self.y_memory = deque(maxlen=mem_size if self.gather_joined_mem else 1)

        # if all actions are visible we need to sum them
        action_compound_size = 0
        if self.joined_actions:
            for ii in range(self.env.n):
                size_act, _ = MAPolicies.get_action_and_comm_actual_size(self.env, ii)
                action_compound_size += size_act

        self.local_learners = []
        for ii in range(self.env.n):
            size_act, _ = MAPolicies.get_action_and_comm_actual_size(self.env, ii)
            self.local_learners.append(
                ModelLearner(environment, ii, action_compound_size if self.joined_actions else size_act,
                             mem_size=mem_size,
                             epochs=epochs,
                             learning_rate=learning_rate,
                             sequence_length=sequence_length,
                             write_tboard=write_tboard,
                             envname=scenario_name))

    # get action from model using random policy
    def get_action(self, obs_n):
        act_n = []
        for ii, ll in enumerate(self.local_learners):
            act_n.append(ll.get_action(obs_n[ii]))
        return act_n

    def get_transition(self, act_n):
        # step environment
        obs_n_next, reward_n, done_n, info_n = self.env.step(act_n)
        # get the real action
        act_n_real = [agent.action.u for agent in self.env.agents]
        return obs_n_next, act_n_real, reward_n, done_n, info_n

    def fill_memory(self):
        # execution loop
        self.x_memory.clear()
        self.y_memory.clear()
        for ll in self.local_learners:
            ll.clear_mem()

        obs_n = env.reset()
        for ii in range(self.mem_size):
            # query for action from each agent's policy
            act_n = self.get_action(obs_n)

            # do a transition
            obs_n_next, act_n_real, reward_n, done_n, info_n = self.get_transition(act_n)

            # TODO is there any done in this environment
            if ((ii % self.sequence_length) == 0 if self.sequence_length > 0 else True) \
                    and random.randrange(self.reset_randomrange) == 1:
                done_n = [True for _ in self.env.agents]

            # save the sample <s, a, r, s'>
            self.x_memory.append((np.array(obs_n).flatten(), np.array(act_n_real).flatten()))
            self.y_memory.append((np.array(reward_n).flatten(), np.array(obs_n_next).flatten()))

            # hand them to the individual learners
            for jj, ll in enumerate(self.local_learners):
                ll.append_to_mem(obs_n[jj],
                                 np.array(act_n_real).flatten() if self.joined_actions else act_n_real[jj],
                                 reward_n[jj],
                                 obs_n_next[jj],
                                 done_n[jj])

            obs_n = obs_n_next

            if any(done_n):
                obs_n = env.reset()

            if self.render:
                env.render()

    def setup_batch_for_RNN(self, batch):
        raise NotImplementedError

    # pick samples randomly from replay memory (with batch_size)
    def train_models(self, minibatch_size=32):
        for ii, ll in enumerate(self.local_learners):
            ll.train_models(minibatch_size=minibatch_size)

    def step_model(self, state, action):
        raise NotImplementedError

    def run(self, rounds=1):
        for e in range(rounds):
            self.fill_memory()
            self.train_models()

    def evaluate(self, environment, do_plots=False):
        raise NotImplementedError


if __name__ == "__main__":
    env_name = 'simple'
    env = make_env2.make_env(env_name)

    canary = MultiAgentModelLearner(env, mem_size=10000, sequence_length=0, scenario_name=env_name, epochs=10)
    canary.run(rounds=1)

    # print('MSE: {}'.format(canary.evaluate(env)))
