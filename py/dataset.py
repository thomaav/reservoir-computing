import numpy as np
import torch


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
