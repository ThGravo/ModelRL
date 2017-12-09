import time
import multiprocessing
import numpy as np
from sklearn.model_selection import ParameterGrid
import gym
import transition_learning_RNN as transition_learning

N_RUNS = 10

grid_params = {
    'sequence_length': [1, 2, 8, 32],
    # 'learning_rate': [0.0005, 0.001],
    'tmodel_dim_multipliers': [(6,), (12, 12), (6, 6), (6, 6, 6)]
    # 'tmodel_activations': [('sigmoid',), ('tanh', 'sigmoid'), ('relu', 'sigmoid'), ('relu', 'tanh', 'sigmoid',)]
}
''',('relu', 'relu'), 
                           ('relu', 'sigmoid', 'tanh'), ('tanh', 'relu', 'sigmoid',),
                           ('relu', 'tanh', 'relu'), ('sigmoid', 'relu', 'sigmoid'), ('tanh', 'sigmoid', 'tanh'),
                           ('relu', 'tanh', 'sigmoid'), ('tanh', 'sigmoid', 'relu'), ('sigmoid', 'tanh', 'relu'),
                           ('relu', 'tanh', 'sigmoid', 'relu'), ('tanh', 'sigmoid', 'tanh', 'relu')]}'''
fixed_params = {
    'data_size': 100000,
    'epochs': 8,
    'tmodel_activations': ('relu', 'sigmoid'),
    'learning_rate': .001
}

grid = list(ParameterGrid(grid_params))
# final_scores = np.zeros(len(grid))
threads = []


def evaluate_single(args):
    index, params = args
    params = {**params, **fixed_params}
    weightedMSE = 0
    weights = [1 / 0.0268, 1 / .000004186, 1 / 0.017888, 1 / 0.0007178]
    env_names = ['LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']
    weights = [1]
    env_names = ['Pong-ram-v4']
    env_names = ['Ant-v1']
    mse_dict = {}
    for env_name, weight in zip(env_names, weights):
        MSEs = []
        for i in range(N_RUNS):
            env = gym.make(env_name)
            tlearner = transition_learning.ModelLearner(env.observation_space, env.action_space, **params)
            tlearner.run(env)
            MSEs.append(tlearner.evaluate(env))
        mse = np.mean(MSEs)
        weightedMSE += weight * mse
        mse_dict[env_name] = mse

    print('Finished evaluating {} with MSEs of {}.'.format(params, mse_dict))
    return weightedMSE


def run():
    start_time = time.time()
    print('About to evaluate {} parameter sets.'.format(len(grid)))
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    final_scores = pool.map(evaluate_single, list(enumerate(grid)))

    sort_idx = np.argsort(final_scores)

    for i in sort_idx:
        print('Parameter set {} achieved score of {}'.format(grid[i], final_scores[i]))

    print('Best parameter set was {} with score of {}'.format(grid[np.argmin(final_scores)], np.min(final_scores)))
    print('Worst parameter set was {} with score of {}'.format(grid[np.argmax(final_scores)], np.max(final_scores)))
    print('Execution time: {} sec'.format(time.time() - start_time))


if __name__ == '__main__':
    run()
