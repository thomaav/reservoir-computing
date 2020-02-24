import matplotlib.pyplot as plt
import torch
import numpy as np

from ESN import ESN


def nrmse(y_predicted, y):
    var = torch.std(y)**2
    error = (y - y_predicted)**2
    return float(torch.sqrt(torch.mean(error) / var))


def nmse(y_predicted, y):
    var = torch.std(y)**2
    error = (y - y_predicted)**2
    return float(torch.mean(error) / var)


def kernel_quality(inputs, esn, ks):
    split_overflow = len(inputs) % ks
    if split_overflow != 0:
        inputs = inputs[:-split_overflow]
    us = np.split(inputs, ks)

    M = [0]*ks
    for i, u in enumerate(us):
        esn(u)
        M[i] = np.array(esn.X[-1])

    return np.linalg.matrix_rank(M)


def memory_capacity(esn):
    # Generated according to «Computational analysis of memory capacity in echo
    # state networks», discarding 100 for transients (washout), using 1100 for
    # training and 1000 for testing the memory capacity.
    inputs = torch.FloatTensor(2200).uniform_(-1, 1)
    washout = inputs[:100]
    u_train = inputs[100:1200]
    u_test = inputs[1200:]
    return esn.memory_capacity(washout, u_train, u_test, plot=False)


def run_esn_experiment(params):
    dataset = params['dataset']
    del params['dataset']

    esn = ESN(**params)
    return evaluate_esn(dataset, esn)


def evaluate_esn(dataset, esn, washout=200, plot=False):
    u_train, y_train, u_test, y_test = dataset
    esn(u_train, y_train)

    y_predicted = esn(u_test)
    _nmse = nmse(y_predicted, y_test[washout:])
    _nrmse = nrmse(y_predicted, y_test[washout:])

    if plot:
        target = y_test[washout:]
        predicted = y_predicted

        plt.plot(target, 'black', label='Target output')
        plt.plot(predicted, 'red', label='Predicted output', alpha=0.5)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0., fancybox=False)

        plt.ylabel('Reservoir output')
        plt.xlabel('Time')

        plt.show()

    return _nrmse


def eval_esn_with_params(dataset, params={}):
    esn = ESN(**params)
    return evaluate_esn(dataset, esn), esn


def evaluate_prediction(y_predicted, y):
    plt.plot(y, 'black', linestyle='dashed')
    plt.plot(y_predicted, 'green')
    plt.show()
