from MDP_learning.from_pixels import dqn_kerasrl_modellearn
import gym


env_name = 'Seaquest-v4'
environment = gym.make(env_name)
num_actions = environment.action_space.n

INPUT_SHAPE = (84, 84)
processor = dqn_kerasrl_modellearn.AtariProcessor(INPUT_SHAPE)
atariCfg = dqn_kerasrl_modellearn.AtariConfig(env_name)
dqn_agent = dqn_kerasrl_modellearn.setupDQN(atariCfg, num_actions, processor)

dqn_kerasrl_modellearn.trainDQN(atariCfg, environment, dqn_agent)