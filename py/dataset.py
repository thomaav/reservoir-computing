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

# Hack to keep the dataset as a global variable from dataset.dataset in other
# modules without passing it explicitly to functions. It is re-initialized
# whenever NARMA is re-run.
dataset = None

def NARMA(sample_len, system_order=10):
    n = system_order

    if n == 10 or n == 20:
        alpha = 0.3
        beta = 0.05
        gamma = 1.5
        delta = 0.1
    elif n == 30:
        # Unchanged for now.
        alpha = 0.3
        beta = 0.05
        gamma = 1.5
        delta = 0.1
    else:
        raise ValueError('Invalid system order for NARMA time series')

    u = torch.rand(sample_len) * 0.5
    y = torch.zeros(sample_len)

    for t in range(n, sample_len):
        y[t] = alpha*y[t-1] + \
               beta*y[t-1]*torch.sum(y[t-n:t]) + \
               gamma*u[t-1]*u[t-n] + \
               delta
        if n != 10:
            y[t] = np.tanh(y[t])

    if not np.isfinite(y).all():
        class DivergentTimeseriesError(Exception):
            pass
        raise DivergentTimeseriesError('Divergent NARMA time series, try again')

    return torch.FloatTensor(u), torch.FloatTensor(y)
