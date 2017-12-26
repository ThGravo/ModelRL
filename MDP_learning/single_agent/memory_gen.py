import gym
import numpy as np
from MDP_learning.single_agent.preprocessing import standardise_memory, make_mem_partial_obs, setup_batch_for_RNN, impute_missing
from MDP_learning.single_agent.dynamics_learning import ModelLearner

# ['Ant-v1', 'LunarLander-v2', 'BipedalWalker-v2', FrozenLake8x8-v0, 'MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']:"Pong-ram-v4"
for env_name in ['Swimmer-v1', 'Hopper-v1']:
    env = gym.make(env_name)
    print(env.observation_space)
    canary = ModelLearner(env.observation_space, env.action_space, partial_obs_rate=0.0, sequence_length=1)
    canary.refill_mem(env)
    memory_arr = np.array(canary.memory)
    print("Saving memory")
    np.save('/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'FULL',
            memory_arr)
    for rate in [0.05, 0.1]:
        make_mem_partial_obs(memory_arr,canary.state_size,rate)
        print("Saving corrupted memory")
        np.save(
            '/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'CORRUPT' + str(rate), memory_arr)
        impute_missing(memory_arr,canary.state_size,MICE)
        print("Saving imputed memory")
        np.save(
            '/home/aocc/code/DL/MDP_learning/save_memory/' + str(env_name) + 'IMPUTED' + str(rate) + 'round' + str(
                round), memory_arr)

