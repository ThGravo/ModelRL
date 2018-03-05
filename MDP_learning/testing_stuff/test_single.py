import MDP_learning.single_agent.dynamics_learning as ml
import gym
import numpy as np

env_name = "BipedalWalker-v2"
env = gym.make(env_name)
observation_space = env.observation_space
action_space = env.action_space

ML = ml.ModelLearner(env_name, observation_space, action_space, partial_obs_rate=0.25, sequence_length=3, epochs=100)
mem = np.load('./MDP_learning/save_memory/{}IMPUTED0.25round0.npy'.format(env_name))
ml.standardise_memory(mem, ML.state_size, ML.action_size)
ML.memory = mem
ML.train_models()
