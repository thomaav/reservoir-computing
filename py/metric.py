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
    inputs = torch.rand(i*ks)*2 - 1

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


def generalization(i, esn, ks):
    # «Connectivity, Dynamics and Memory in Reservoir Computing with Binary and
    # Analog Neurons». Footnote 5.
    inputs = np.random.rand(i*ks)*2 - 1

    split_overflow = len(inputs) % ks
    if split_overflow != 0:
        inputs = inputs[:-split_overflow]
    us = np.split(inputs, ks)

    # Generalization part: set the last fifth of the input stream to always be
    # the same.
    gen_length = i // 5
    gen_seq = us[0][-gen_length:]
    for u in us:
        np.put(u, np.arange(-gen_length, 0), gen_seq)
    us = torch.FloatTensor(us)

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


def esn_nrmse(params, attr=[]):
    dataset = ds.dataset
    esn = ESN.ESN(**params)
    esn_attributes = [getattr(esn, _attr) for _attr in attr]
    return evaluate_esn(dataset, esn), esn_attributes


def esn_mc(params, attr=[]):
    esn = ESN.ESN(**params)
    esn_attributes = [getattr(esn, _attr) for _attr in attr]
    return memory_capacity(esn), esn_attributes


def esn_kq(params, attr=[]):
    esn = ESN.ESN(**params)
    esn_attributes = [getattr(esn, _attr) for _attr in attr]
    return kernel_quality(20, esn, esn.hidden_nodes), esn_attributes


def esn_gen(params, attr=[]):
    esn = ESN.ESN(**params)
    esn_attributes = [getattr(esn, _attr) for _attr in attr]
    return generalization(20, esn, esn.hidden_nodes), esn_attributes


def evaluate_esn(dataset, esn, plot=False, plot_range=None, show=True):
    washout = esn.washout

    u_train, y_train, u_test, y_test = dataset
    esn(u_train, y_train)

    y_predicted = esn(u_test)
    _nmse = nmse(y_predicted, y_test[washout:])
    _nrmse = nrmse(y_predicted, y_test[washout:])

    if plot:
        if plot_range is not None:
            i, j = plot_range[0], plot_range[1]
            target = y_test[washout+i:washout+j]
            predicted = y_predicted[i:j]
        else:
            target = y_test[washout:]
            predicted = y_predicted

        plt.plot(target, 'black', label='Target output')
        plt.plot(predicted, 'black', label='Predicted output', linestyle='dashed')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0., fancybox=False)

        plt.ylabel('Output')
        plt.xlabel('Time step')

        if show:
            plt.show()

    return _nrmse
