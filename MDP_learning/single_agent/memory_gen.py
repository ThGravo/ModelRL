import gym
import numpy as np
from MDP_learning.single_agent.preprocessing import make_mem_partial_obs, impute_missing, standardise_memory
from MDP_learning.single_agent.dynamics_learning import ModelLearner
from fancyimpute import MICE

# ['Ant-v1', 'LunarLander-v2', 'BipedalWalker-v2', FrozenLake8x8-v0, 'MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']:"Pong-ram-v4"
for env_name in ['BipedalWalker-v2',
                    #'Swimmer-v1',
                    #'Hopper-v1',
                 ]:
    #env = gym.make(env_name)
    #print(env.observation_space)
    #canary = ModelLearner(env_name, env.observation_space, env.action_space, partial_obs_rate=0.0, sequence_length=0)
    #canary.refill_mem(env)
    #memory_arr = np.array(canary.memory)
    #print("Saving memory")
    #np.save('/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'FULL',
    #        memory_arr)
    memory_arr = np.load('/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'FULL.npy')
    env = gym.make(env_name)
    print(env.observation_space)
    for rate in [0.25, 0.50, 0.75]:
        print('Corrupting memory')
        make_mem_partial_obs(memory_arr,24,rate)
        print("Saving corrupted memory")
        np.save(
            '/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'CORRUPT' + str(rate), memory_arr)
        print('Imputing missing values')
        impute_missing(memory_arr,24,MICE)
        print("Saving imputed memory")
        np.save(
            '/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'IMPUTED' + str(rate), memory_arr)



