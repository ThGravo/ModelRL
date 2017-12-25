import numpy as np
import multiagent.policy


# no communication
class RandomPolicy(multiagent.policy.Policy):
    def __init__(self, env, agent_index):
        super(RandomPolicy, self).__init__()
        self.env = env
        self.agent_index = agent_index

    def action(self, obs):
        # self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        # self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        a_space = self.env.action_space[self.agent_index]

        if self.env.discrete_action_input:
            u = np.random.randint(a_space.n)
        else:
            u = np.zeros(5)  # one-hot-encoded
            if self.env.discrete_action_space:
                act = np.random.uniform(low=0, high=1)
            else:
                act = np.random.uniform(low=a_space.low, high=a_space.high)
            u[np.random.randint(a_space.n)] = act

        return np.concatenate([u, np.zeros(self.env.world.dim_c)])
