import time
import multiprocessing
import numpy as np
from sklearn.model_selection import ParameterGrid

from n1try import qcartpole

N_RUNS = 10

grid_params = {
    'min_alpha': [0.1, 0.2, 0.5],
    'min_epsilon': [0.0, 0.01, 0.1, 0.2],
    'buckets': [(1, 1, 6, 3), (1, 1, 3, 6), (1, 1, 6, 12), (1, 1, 12, 6), (1, 1, 4, 4)],
    'ada_divisor': [25, 50, 100],
    'gamma': [1.0, 0.99, 0.9]
}

fixed_params = {
    'quiet': True
}

grid = list(ParameterGrid(grid_params))
final_scores = np.zeros(len(grid))
threads = []

def evaluate_single(args):
    index, params = args
    print('Evaluating params: {}'.format(params))
    params = {**params, **fixed_params}

    scores = []
    for i in range(N_RUNS):
        solver = qcartpole.QCartPoleSolver(**params)
        score = solver.run()
        scores.append(score)

    score = np.mean(scores)
    print('Finished evaluating set {} with score of {}.'.format(index, score))
    return score

def run():
    start_time = time.time()
    print('About to evaluate {} parameter sets.'.format(len(grid)))
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    final_scores = pool.map(evaluate_single, list(enumerate(grid)))

    print('Best parameter set was {} with score of {}'.format(grid[np.argmin(final_scores)], np.min(final_scores)))
    print('Worst parameter set was {} with score of {}'.format(grid[np.argmax(final_scores)], np.max(final_scores)))
    print('Execution time: {} sec'.format(time.time() - start_time))

if __name__ == '__main__':
    run()