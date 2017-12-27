import numpy as np
import multiagent.policy
from gym import spaces


def get_action_and_comm_size(action_space, agent):
    size_act = 0
    size_com = 0
    if isinstance(action_space, spaces.MultiDiscrete):
        size = action_space.high - action_space.low + 1
        size_act = size[0] if agent.movable else 0
        size_com = size[1] if not agent.silent else 0
    elif isinstance(action_space, spaces.Discrete):
        size_act = action_space.n if agent.movable else 0
        size_com = action_space.n if not agent.silent else 0
    elif isinstance(action_space, spaces.Box):
        size_act = sum(action_space.shape) if agent.movable else 0
        size_com = sum(action_space.shape) if not agent.silent else 0
    elif isinstance(action_space, spaces.Tuple):
        assert len(action_space.spaces) == 2
        # TODO duplicate from above - reduce code duplication
        if isinstance(action_space.spaces[0], spaces.Discrete):
            size_act = action_space.spaces[0].n if agent.movable else 0
        elif isinstance(action_space.spaces[0], spaces.Box):
            size_act = sum(action_space.spaces[0].shape) if agent.movable else 0

        if isinstance(action_space.spaces[1], spaces.Discrete):
            size_com = action_space.spaces[1].n if not agent.silent else 0
        elif isinstance(action_space.spaces[1], spaces.Box):
            size_com = sum(action_space.spaces[1].shape) if not agent.silent else 0
        assert False  # not sure if that's correct - as there is no example of it
    else:
        raise NotImplementedError()
    return size_act, size_com


# no communication
class RandomPolicy(multiagent.policy.Policy):
    def __init__(self, env, agent_index):
        super(RandomPolicy, self).__init__()
        self.env = env
        self.agent_index = agent_index

    def action(self, obs):
        agent = self.env.agents[self.agent_index]
        # environment.py
        # self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        # self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        action_space = self.env.action_space[self.agent_index]

        size_act, size_com = get_action_and_comm_size(action_space, agent)

        u = np.array([])
        if agent.movable:
            # physical action
            if self.env.discrete_action_input:
                u = np.random.randint(size_act)
            else:
                if self.env.discrete_action_space:
                    # one-hot encoded
                    u = np.zeros(size_act)
                    u[np.random.randint(size_act)] = 1.0
                else:
                    u = np.random.uniform(low=action_space.low, high=action_space.high, size=size_act)

        c = np.array([])
        if not agent.silent:
            # communication action
            if self.env.discrete_action_input:
                '''c = np.zeros(self.env.world.dim_c)
                c[action[0]] = 1.0'''
                c = np.random.randint(size_com)
            else:
                '''c = action[0]'''
                c = np.random.uniform(low=0.0, high=1.0, size=size_com)

        return np.concatenate([u, c])
