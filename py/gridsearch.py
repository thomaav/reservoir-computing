import numpy as np
from sklearn.model_selection import ParameterGrid

from ESN import ESN
from metric import evaluate_esn, eval_esn_with_params


def evaluate_esn_1d(dataset, params, runs_per_iteration=1):
    if len(params.keys()) != 1:
        class InvalidParameterDictException(Exception):
            pass
        raise InvalidParameterDictException('1d grid search requires exactly 1 parameter lists')

    output = []
    param_grid = ParameterGrid(params)
    p1 = sorted(params.keys())[0]

    for params in param_grid:
        print(params)
        nrmses = []

        for i in range(runs_per_iteration):
            esn = ESN(**params)
            nrmses.append(evaluate_esn(dataset, esn))

        output.append(np.mean(nrmses))

    return output


def evaluate_esn_2d(dataset, params, runs_per_iteration=1):
    if len(params.keys()) != 2:
        class InvalidParameterDictException(Exception):
            pass
        raise InvalidParameterDictException('2d grid search requires exactly 2 parameter lists')

    sorted_params = sorted(params.keys())
    p1 = sorted_params[0]
    p2 = sorted_params[1]
    n_p1, n_p2 = len(params[p1]), len(params[p2])
    output = [[] for _ in range(n_p1)]

    param_grid = ParameterGrid(params)
    for i, params in enumerate(param_grid):
        print(params)
        p1_i = i // n_p2

        nrmses = []
        for i in range(runs_per_iteration):
            nrmse = eval_esn_with_params(dataset, params)
            nrmses.append(nrmse)
        output[p1_i].append(np.mean(nrmses))

    return output
