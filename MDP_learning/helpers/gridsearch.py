import time
import multiprocessing
import numpy as np
from sklearn.model_selection import ParameterGrid
import gym
import transition_learning

N_RUNS = 10

grid_params = {
    'learning_rate': [0.001, 0.005, 0.01, 0.1],
    'tmodel_dim_multipliers': [(1, 1), (3, 3), (6, 3), (12, 3), (6, 6), (1, 1, 1), (3, 3, 3), (1, 1, 1, 1),
                               (3, 3, 3, 3)],
    'tmodel_activations': [('relu', 'relu'), ('sigmoid', 'sigmoid'), ('tanh', 'tanh'),
                           ('relu', 'sigmoid', 'tanh'), ('tanh', 'relu', 'sigmoid',),
                           ('relu', 'tanh', 'relu'), ('sigmoid', 'relu', 'sigmoid'), ('tanh', 'sigmoid', 'tanh'),
                           ('relu', 'tanh', 'sigmoid'), ('tanh', 'sigmoid', 'relu'), ('sigmoid', 'tanh', 'relu'),
                           ('relu', 'tanh', 'sigmoid', 'relu'), ('tanh', 'sigmoid', 'tanh', 'relu')]
}

fixed_params = {
    'action_size': 1
}

grid = list(ParameterGrid(grid_params))
final_scores = np.zeros(len(grid))
threads = []


def evaluate_single(args):
    index, params = args
    print('Evaluating params: {}'.format(params))
    params = {**params, **fixed_params}
    weightedMSE = 0
    weights = [1 / 0.0268, 1 / .000004186, 1 / 0.017888, 1 / 0.0007178]
    env_names = ['LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1', 'CartPole-v1'];
    for env_name, weight in zip(env_names, weights):
        MSEs = []
        for i in range(N_RUNS):
            env = gym.make(env_name)
            tlearner = transition_learning.ModelLearner(sum(env.observation_space.shape),
                                                      env.action_space.n,
                                                      **params)
            tlearner.run(env)
            MSEs.append(tlearner.evaluate(env))

        mse = np.mean(MSEs)
        print('Finished evaluating env {} with MSE of {}.'.format(env_name, mse))
        weightedMSE += weight * mse

    return weightedMSE


def run():
    start_time = time.time()
    print('About to evaluate {} parameter sets.'.format(len(grid)))
    pool = multiprocessing.Pool(processes=8) # multiprocessing.cpu_count())
    final_scores = pool.map(evaluate_single, list(enumerate(grid)))

    print('Best parameter set was {} with score of {}'.format(grid[np.argmin(final_scores)], np.min(final_scores)))
    print('Worst parameter set was {} with score of {}'.format(grid[np.argmax(final_scores)], np.max(final_scores)))
    print('Execution time: {} sec'.format(time.time() - start_time))


if __name__ == '__main__':
    run()
