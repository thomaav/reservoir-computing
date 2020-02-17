import numpy as np
import torch


def cache_dataset(dataset):
    u_train, y_train, u_test, y_test = dataset

    np.save('dataset/u_train.npy', u_train)
    np.save('dataset/y_train.npy', y_train)
    np.save('dataset/u_test.npy', u_test)
    np.save('dataset/y_test.npy', y_test)


def load_dataset():
    u_train = torch.FloatTensor(np.load('dataset/u_train.npy'))
    y_train = torch.FloatTensor(np.load('dataset/y_train.npy'))
    u_test = torch.FloatTensor(np.load('dataset/u_test.npy'))
    y_test = torch.FloatTensor(np.load('dataset/y_test.npy'))
    return [u_train, y_train, u_test, y_test]


def NARMA(sample_len, system_order=10):
    if system_order != 10:
        raise NotImplementedError('NARMA only supported for system order of 10')

    alpha = 0.3
    beta = 0.05
    gamma = 1.5
    delta = 0.1

    u = torch.rand(sample_len) * 0.5
    y = torch.zeros(sample_len)

    for t in range(10, sample_len):
        y[t] = alpha*y[t-1] + \
               beta*y[t-1]*torch.sum(y[t-10:t]) + \
               gamma*u[t-1]*u[t-10] + \
               delta

    if not np.isfinite(y).all():
        class DivergentTimeseriesError(Exception):
            pass
        raise DivergentTimeseriesError('Divergent NARMA time series, try again')

    return torch.FloatTensor(u), torch.FloatTensor(y)
