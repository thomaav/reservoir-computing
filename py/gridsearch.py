import numpy as np
from sklearn.model_selection import ParameterGrid

from ESN import ESN
from metric import evaluate_esn, eval_esn_with_params
from util import snr


def evaluate_esn_1d(dataset, params, runs_per_iteration=1, test_snrs=None):
    if len(params.keys()) != 1:
        class InvalidParameterDictException(Exception):
            pass
        raise InvalidParameterDictException('1d grid search requires exactly 1 parameter lists')

    output = []
    param_grid = ParameterGrid(params)
    p1 = sorted(params.keys())[0]

    # We need the SNR of u/v.
    u_train, _, u_test, _ = dataset

    for params in param_grid:
        print(params)
        nrmses = []
        test_snrs_t = []

        for i in range(runs_per_iteration):
            esn = ESN(**params)
            nrmses.append(evaluate_esn(dataset, esn))
            if test_snrs is not None:
                test_snrs_t.append(snr(u_test.var(), esn.v.var()))
                v = esn.v

        output.append(np.mean(nrmses))
        if test_snrs is not None:
            test_snrs.append(np.mean(test_snrs_t))

    return output


def evaluate_esn_2d(dataset, params, runs_per_iteration=1, test_snrs=None):
    if len(params.keys()) != 2:
        class InvalidParameterDictException(Exception):
            pass
        raise InvalidParameterDictException('2d grid search requires exactly 2 parameter lists')

    # We need the SNR of u/v.
    u_train, _, u_test, _ = dataset

    sorted_params = sorted(params.keys())
    p1 = sorted_params[0]
    p2 = sorted_params[1]
    n_p1, n_p2 = len(params[p1]), len(params[p2])
    nrmse_output = [[] for _ in range(n_p1)]
    std_output = [[] for _ in range(n_p1)]
    for i in range(n_p1):
        if test_snrs is not None:
            test_snrs.append([])

    param_grid = ParameterGrid(params)
    for i, params in enumerate(param_grid):
        print(params)
        p1_i = i // n_p2

        nrmses = []
        test_snrs_t = []
        for i in range(runs_per_iteration):
            # (TODO): Change this to return everything instead of passing lists
            # to this grid function.
            nrmse, esn = eval_esn_with_params(dataset, params)
            nrmses.append(nrmse)
            if test_snrs is not None:
                test_snrs_t.append(snr(u_test.var(), esn.v.var()))
                v = esn.v

        nrmse_output[p1_i].append(np.mean(nrmses))
        std_output[p1_i].append(np.std(nrmses))
        if test_snrs is not None:
            test_snrs[p1_i].append(np.mean(test_snrs_t))

    # (TODO): Return a dict of different metrics that includes SNR.
    return nrmse_output, std_output
