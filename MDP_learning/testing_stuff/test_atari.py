from MDP_learning.from_pixels import dqn_kerasrl_modellearn
import gym


env_name = 'SeaquestDeterministic-v4'
environment = gym.make(env_name)
num_actions = environment.action_space.n

INPUT_SHAPE = (84, 84)
processor = dqn_kerasrl_modellearn.AtariProcessor(INPUT_SHAPE)
dqn_agent, hidden_state_size = dqn_kerasrl_modellearn.setupDQN(num_actions, processor)

dqn_kerasrl_modellearn.trainDQN(environment, dqn_agent)