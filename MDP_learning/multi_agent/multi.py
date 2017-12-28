from MDP_learning.multi_agent import make_env2
import MDP_learning.multi_agent.policies as MAPolicies

from MDP_learning.helpers import build_models
from collections import deque
import random
import numpy as np
# matplotlib.use('GTK3Cairo', warn=False, force=True)
import gym

from MDP_learning.helpers.logging_model_learner import LoggingModelLearner


class ModelLearner(LoggingModelLearner):
    def __init__(self, environment, policy, observation_space, action_space,
                 state_size=None,action_size=None,action_num=None,
                 mem_size=3000, epochs=4, learning_rate=.001,
                 sequence_length=0, write_tboard=True, agent_id=None):

        super().__init__(environment,sequence_length,
                         write_tboard=write_tboard,
                         out_dir_add='_agentID{}'.format(agent_id) if agent_id is not None else None)

        self.learning_rate = learning_rate
        self.net_train_epochs = epochs
        self.mem_size = mem_size
        self.memory = deque(maxlen=self.mem_size)

        self.policy = policy

        # get size of state and action from environment
        self.state_size = sum(observation_space.shape) if state_size is None else state_size

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_size = 1 if action_size is None else action_size
            self.action_num = action_space.n if action_num is None else action_num
        elif isinstance(action_space, gym.spaces.Box):
            self.action_size = sum(action_space.shape) if action_size is None else action_size
            self.action_num = 0 if action_num is None else action_num
        else:
            raise ValueError("The action_space is of type:"
                             " {} - which is not supported!".format(type(action_space)))


    # get action from model using random policy
    def get_action(self, state):
        #return self.env.action_space.sample()
        return self.policy.action()

    def run(self, environment, rounds=1):
        for e in range(rounds):
            self.refill_mem(environment)
            self.train_models()


class MultiAgentModelLearner(LoggingModelLearner):
    def __init__(self, environment, mem_size=3000, epochs=4, learning_rate=.001,
                 sequence_length=0, write_tboard=True, agent_id=None                 ):
        super().__init__(environment, sequence_length,
                         write_tboard=write_tboard,
                         out_dir_add='_agentID{}'.format(agent_id) if agent_id is not None else None)

        self.local_learners = []

        for i in range(self.n):
            policies = [MAPolicies.RandomPolicy(env, i) for i in range(env.n)]
            self.local_learners.append(ModelLearner(environment))

    # get action from model using random policy
    def get_action(self, obs_n):
        actions = []
        for i in range(self.n):
            act = np.array([random.uniform(-2, 2) for j in range(env.action_space[i].n)])
            actions.append(act)
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        return actions

    # pick samples randomly from replay memory (with batch_size)
    def train_models(self, minibatch_size=32):

    def step(self, state, action):
        raise NotImplementedError

    def refill_mem(self, environment):
        state = environment.reset()
        self.memory.clear()
        for i in range(self.data_size):
            # get action for the current state and go one step in environment
            action = self.get_action(state, environment)
            next_state, reward, done, info = environment.step(action)

            # save the sample <s, a, r, s'> to the replay memory
            self.memory.append((state, action, reward, next_state, done * 1))

            if done:
                state = environment.reset()
            else:
                state = next_state

    def setup_batch_for_RNN(self, batch):
        raise NotImplementedError

    def run(self, environment, rounds=1):
        for e in range(rounds):
            self.refill_mem(environment)
            self.train_models()

    def evaluate(self, environment, do_plots=False):
        raise NotImplementedError


if __name__ == "__main__":
    env_name = 'simple_spread'
    env = make_env2.make_env(env_name)

    canary = MultiAgentModelLearner(env, sequence_length=1)
    canary.run(env, rounds=8)

    # print('MSE: {}'.format(canary.evaluate(env)))
