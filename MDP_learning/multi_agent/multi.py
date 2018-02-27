from MDP_learning.multi_agent import make_env2
import MDP_learning.multi_agent.policies as MAPolicies
import MDP_learning.multi_agent.ModelLearner as ModelLearner
from MDP_learning.helpers.logging_model_learner import LoggingModelLearner

import random
import numpy as np
from collections import deque


class MultiAgentModelLearner(LoggingModelLearner):
    def __init__(self, environment, mem_size=3000, epochs=4, learning_rate=.001,
                 sequence_length=0, write_tboard=True, scenario_name=None, net_depth=2):
        super().__init__(environment, sequence_length,
                         write_tboard=write_tboard,
                         out_dir_add='scenario_name{}'.format(scenario_name) if scenario_name is not None else None)
        self.render = False
        self.joined_actions = False
        self.gather_joined_mem = False
        self.random_resets = True
        # how likely a random reset is (1 is resetting always)
        self.reset_randomrange = 3 if sequence_length > 0 else int(mem_size / 100) + 2
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
                ModelLearner.ModelLearner(environment, ii, action_compound_size if self.joined_actions else size_act,
                                          mem_size=mem_size,
                                          epochs=epochs,
                                          learning_rate=learning_rate,
                                          sequence_length=sequence_length,
                                          write_tboard=write_tboard,
                                          envname=scenario_name,
                                          net_depth=net_depth))

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

        obs_n = self.env.reset()
        for ii in range(self.mem_size):
            # query for action from each agent's policy
            act_n = self.get_action(obs_n)

            # do a transition
            obs_n_next, act_n_real, reward_n, done_n, info_n = self.get_transition(act_n)

            if ((ii % self.sequence_length) == 0 if self.sequence_length > 0 else True) \
                    and self.random_resets and random.randrange(self.reset_randomrange) == 0:
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
                obs_n = self.env.reset()

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
            self.train_models(64)

    def evaluate(self, environment, do_plots=False):
        raise NotImplementedError


if __name__ == "__main__":
    if False:
        s = 50
        env_name = 'simple'
        env = make_env2.make_env(env_name)
        canary = MultiAgentModelLearner(env, scenario_name=env_name,
                                        mem_size=10000 * (s + 1),
                                        sequence_length=s,
                                        epochs=10,
                                        net_depth=2)
        canary.run(rounds=1)
    else:
        for env_name in ['simple', 'simple_spread', 'simple_push']:
            for s in [100, 30, 0, 3, 10]:
                env = make_env2.make_env(env_name)
                for nd in [2, 3, 1]:
                    print('Running Env {} with Seqlen {} and NetDepth {}'.format(env_name, s, nd))
                    canary = MultiAgentModelLearner(env, scenario_name=env_name,
                                                    mem_size=300000 * (s + 1),
                                                    sequence_length=s,
                                                    epochs=100,
                                                    net_depth=nd,
                                                    learning_rate=.001)
                    canary.run(rounds=1)
