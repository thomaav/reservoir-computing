import matplotlib.pyplot as plt
import torch
import numpy as np

import dataset as ds
import ESN


def nrmse(y_predicted, y):
    var = torch.std(y)**2
    error = (y - y_predicted)**2
    return float(torch.sqrt(torch.mean(error) / var))


def nmse(y_predicted, y):
    var = torch.std(y)**2
    error = (y - y_predicted)**2
    return float(torch.mean(error) / var)


# Also used: Advances in Unconvential Computing: Volume 1: Theory. Andrew
# Adamatzky, 22.6.1.1.


def kernel_quality(i, esn, ks):
    # «Connectivity, Dynamics and Memory in Reservoir Computing with Binary and
    # Analog Neurons».
    inputs = torch.rand(i*ks)

    split_overflow = len(inputs) % ks
    if split_overflow != 0:
        inputs = inputs[:-split_overflow]
    us = np.split(inputs, ks)

    M = [0]*ks
    for i, u in enumerate(us):
        esn(u, kq=True)
        M[i] = np.array(esn.X[-1])

    kq = np.linalg.matrix_rank(M)
    return kq


def memory_capacity(esn):
    # Generated according to «Computational analysis of memory capacity in echo
    # state networks», discarding 100 for transients (washout), using 1100 for
    # training and 1000 for testing the memory capacity.
    torch.manual_seed(0)
    inputs = torch.FloatTensor(2200).uniform_(-1, 1)
    washout = inputs[:100]
    u_train = inputs[100:1200]
    u_test = inputs[1200:]
    return esn.memory_capacity(washout, u_train, u_test, plot=False)


def esn_nrmse(params):
    dataset = ds.dataset
    esn = ESN.ESN(**params)
    return evaluate_esn(dataset, esn)


def esn_mc(params):
    esn = ESN.ESN(**params)
    return memory_capacity(esn)


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
