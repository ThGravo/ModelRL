import gym
import numpy as np
from MDP_learning.single_agent.preprocessing import make_mem_partial_obs, impute_missing, standardise_memory
from MDP_learning.single_agent.dynamics_learning import ModelLearner
from fancyimpute import MICE

for env_name in ['Swimmer-v1',
                    #'BipedalWalker-v2',
                    #'Hopper-v1',
                 ]:
    '''
    Use the following to generate memories from scratch:
    
    env = gym.make(env_name)
    print(env.observation_space)
    canary = ModelLearner(env_name, env.observation_space, env.action_space, partial_obs_rate=0.0, sequence_length=0)
    canary.refill_mem(env)
    memory_arr = np.array(canary.memory)
    print("Saving memory")
    np.save('/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'FULL',
            memory_arr)
        
    '''
    # Load full memory to corrupt
    memory_arr = np.load('/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'FULL.npy')
    # partial observability rates
    for rate in [0.25,0.50,0.75]:
            for round in [0,1,2,3,4]:
                print('Corrupting memory')
                mem = memory_arr[round*1000000:round*1000000 + 1000000,...]
                make_mem_partial_obs(mem,24,rate)
                print("Saving corrupted memory")
                np.save(
                    '/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'CORRUPT' + str(rate), memory_arr)
                print('Imputing missing values')
                impute_missing(mem,24,MICE)
                print("Saving imputed memory")
                np.save(
                    '/home/aocc/code/DL/MDP_learning/save_memory1/' + str(env_name) + 'IMPUTED' + str(rate) + 'round' + str(round), mem)
                memory_arr = np.load('/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'FULL.npy')



