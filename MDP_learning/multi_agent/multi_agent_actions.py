from MDP_learning.multi_agent import make_env
import numpy as np

env_name = 'simple_spread'
env = make_env.make_env(env_name)
env.agents = env.world.policy_agents
print(env.action_space)

x = env.step([np.array([-1,1,-2,0,1]),np.array([0,0,1,0,0]),np.array([1,0,0,0,0])])
print(x)
print(env.discrete_action_space)

