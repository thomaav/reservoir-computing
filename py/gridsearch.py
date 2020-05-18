import numpy as np
import pandas as pd
import os
import sys
import multiprocessing as mp
import tqdm
import cProfile
from sklearn.model_selection import ParameterGrid
from itertools import product, islice, chain
from functools import partial

from ESN import ESN
from metric import evaluate_esn
from util import snr


def dict_chunks(data, n):
    size = len(data) // n
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def list_chunks(lst, n):
    size = len(lst) // n
    rem = len(lst) % n

    chunks = []
    for i in range(n):
        chunks.append(lst[i*size:(i+1)*size])

    for i in range(rem):
        chunks[i%n].append(lst[-(i+1)])

    return chunks


def _experiment(f, param_names, param_values, esn_attributes=[], affinity=0, runs=10):
    os.sched_setaffinity(0, {affinity})

    results = []
    for experiment in param_values:
        _params = {pn: pv for pn, pv in zip(param_names, experiment)}
        print(_params)

        for i in range(runs):
            result, attr = f(_params, esn_attributes)
            results.append([result] + list(_params.values()) + attr)

    return results


def experiment(f, params, runs=10, esn_attributes=[]):
    param_names = [l[0] for l in params.items()]
    param_values = list(product(*[l[1] for l in params.items()]))

    cpus = mp.cpu_count()-1 or 1
    pool = mp.Pool(processes=cpus)

    try:
        results = []
        chunked_param_values = list_chunks(param_values, cpus)
        for affinity, chunk in enumerate(chunked_param_values):
            results.append(pool.apply_async(
                _experiment, (f, param_names, chunk, esn_attributes, affinity, runs)
            ))
    except KeyboardInterrupt:
        pool.terminate()
        exit("KeyboardInterrupt received, shut down experiment processes")

    pool.close()
    pool.join()

    results = list(chain(*[r.get() for r in results]))
    column_names = [f.__name__, *param_names, *esn_attributes]
    df = pd.DataFrame(results, columns=column_names)
    return df


def load_experiment(path):
    if os.path.isfile(path):
        return pd.read_pickle(path)
    raise FileNotFoundError
