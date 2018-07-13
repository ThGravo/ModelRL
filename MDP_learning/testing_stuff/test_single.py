import MDP_learning.single_agent.dynamics_learning as ml
import gym

env_name = "BipedalWalker-v2"
env = gym.make(env_name)
observation_space = env.observation_space
action_space = env.action_space

ML = ml.ModelLearner(env_name, observation_space, action_space, partial_obs_rate=0.0, sequence_length=4, epochs=100)
ML.run(env)
