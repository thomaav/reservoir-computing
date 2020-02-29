import numpy as np
import pandas as pd
import os
import multiprocessing
from sklearn.model_selection import ParameterGrid
from itertools import product

from ESN import ESN
from metric import evaluate_esn
from util import snr


def experiment(f, params, runs=10):
    dim = len(params)
    param_names = [l[0] for l in params.items()]
    param_values = [l[1] for l in params.items()]
    results = []

    for experiment in product(*param_values):
        _params = {pn: pv for pn, pv in zip(param_names, experiment)}
        print(_params)

        for i in range(runs):
            # (TODO): append gets really slow eventually.
            result = f(_params)
            results.append([result] + list(_params.values()))

    column_names = [f.__name__, *param_names]
    df = pd.DataFrame(results, columns=column_names)
    return df


def load_experiment(path):
    if os.path.isfile(path):
        return pd.read_pickle(path)
    raise FileNotFoundError
