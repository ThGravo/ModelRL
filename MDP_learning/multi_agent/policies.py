import numpy as np
import multiagent.policy
from gym import spaces


def discrete_space_sample(n):
    a = np.zeros(n)
    i = np.random.randint(n + 1)
    # no action is also possible
    if i < n:
        a[i] = 1
    return a


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
        smpl = action_space.sample()

        size_act = 0
        size_com = 0
        if isinstance(action_space, spaces.MultiDiscrete):
            size = action_space.high - action_space.low + 1
            size_act = size[0]
            size_com = self.env.world.dim_c
        elif isinstance(action_space, spaces.Discrete):
            size_act = action_space.n if agent.movable else 0
            size_com = 0 if agent.silent else self.env.world.dim_c

        u = np.array([])
        if agent.movable:
            # physical action
            if self.env.discrete_action_input:
                u = np.random.randint(size_act)
                '''
                u = np.zeros(self.env.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
                '''
            else:
                '''if self.env.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0'''
                if self.env.discrete_action_space:
                    '''u[0] += action[0][1] - action[0][2]
                    u[1] += action[0][3] - action[0][4]'''
                    # one-hot encoded
                    u = np.zeros(size_act)
                    u[np.random.randint(size_act)] = 1.0
                else:
                    u = np.random.uniform(low=action_space.low, high=action_space.high, size=action_space.shape)

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

        return np.concatenate([u,c])


        if self.env.discrete_action_input:
            return smpl
        else:
            u = np.zeros(self.env.world.dim_p * 2 + 1)
            if self.env.discrete_action_space:
                u = np.zeros(self.env.world.dim_p * 2 + 1)
                #u = np.random.uniform(low=action_space.low, high=action_space.high, size=action_space.shape)
                u[smpl[0]] = 1.0
            else:
                u = np.zeros(self.env.world.dim_p)
                u[smpl[0]] = 1.0
            return np.concatenate([u, np.zeros(self.env.world.dim_c)])

        if isinstance(action_space, spaces.Discrete):
            s = discrete_space_sample(action_space.n)
        if isinstance(action_space, spaces.Box):
            s = np.random.uniform(low=action_space.low, high=action_space.high, size=action_space.shape)

        if isinstance(action_space, spaces.MultiDiscrete):
            s = action_space.sample()
        if isinstance(action_space, spaces.Tuple):
            s = action_space.sample()

        if self.env.discrete_action_input:
            u = np.random.randint(action_space.n)
        else:
            u = np.zeros(5)  # one-hot-encoded TODO hard-coded 5
            if self.env.discrete_action_space:
                act = np.random.uniform(low=0, high=1)
            else:
                act = np.random.uniform(low=action_space.low, high=action_space.high)
            u[np.random.randint(action_space.n)] = act

        if agent.movable:
            if agent.silent:
                return u
            else:
                return np.concatenate([u, np.zeros(self.env.world.dim_c)])
