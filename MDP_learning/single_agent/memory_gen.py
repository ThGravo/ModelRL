import gym
import numpy as np
from MDP_learning.single_agent.preprocessing import make_mem_partial_obs, impute_missing
from MDP_learning.single_agent.dynamics_learning import ModelLearner
from fancyimpute import MICE

# ['Ant-v1', 'LunarLander-v2', 'BipedalWalker-v2', FrozenLake8x8-v0, 'MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']:"Pong-ram-v4"
for env_name in ['HalfCheetah-v1'
                #'Swimmer-v1',
                #'Hopper-v1'
                 ]:
    env = gym.make(env_name)
    print(env.observation_space)
    canary = ModelLearner(env_name, env.observation_space, env.action_space, partial_obs_rate=0.0, sequence_length=0)
    canary.refill_mem(env)
    memory_arr = np.array(canary.memory)
    print("Saving memory")
    np.save('/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'FULL',
            memory_arr)
    for rate in [0.01, 0.05, 0.1]:
        make_mem_partial_obs(memory_arr,canary.state_size,rate)
        print("Saving corrupted memory")
        np.save(
            '/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'CORRUPT' + str(rate), memory_arr)
        impute_missing(memory_arr,canary.state_size,MICE)
        print("Saving imputed memory")
        np.save(
            '/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'IMPUTED' + str(rate) + 'round' + str(
                round), memory_arr)

