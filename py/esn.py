import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def spectral_radius(w):
    return torch.max(torch.abs(torch.eig(w)[0])).item()


def nrmse(y_predicted, y):
    var = torch.std(y)**2
    error = (y - y_predicted)**2
    return float(torch.sqrt(torch.mean(error) / var))


def evaluate_prediction(y_predicted, y):
    plt.plot(y, 'black', linestyle='dashed')
    plt.plot(y_predicted, 'green')
    plt.show()


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


class ESN(nn.Module):
    def __init__(self):
        super(ESN, self).__init__()

        self.hidden_nodes = 200
        self.spectral_radius = 0.9
        self.f = torch.tanh
        self.w_res_sparsity = 0.2
        self.washout = 200

        w_res = torch.rand(self.hidden_nodes, self.hidden_nodes) - 0.5
        w_res[torch.rand(self.hidden_nodes, self.hidden_nodes) > self.w_res_sparsity] = 0.0
        w_res *= self.spectral_radius / spectral_radius(w_res)
        w_in = torch.rand(self.hidden_nodes) - 0.5
        w_out = torch.ones(self.hidden_nodes)

        self.register_buffer('w_res', w_res)
        self.register_buffer('w_in', w_in)
        self.register_buffer('w_out', w_out)


    def forward(self, u, y=None):
        timeseries_len = u.size()[0]
        X = torch.zeros(timeseries_len, self.hidden_nodes)
        x = torch.zeros(self.hidden_nodes)

        for t in range(timeseries_len):
            u_t = self.w_in * u[t]
            x_t = self.w_res.mv(x)
            x = self.f(u_t + x_t)
            X[t] = x

        X = X[self.washout:]
        y = y[self.washout:] if y is not None else y

        if y is not None:
            self.w_out = torch.mv(torch.pinverse(X), y)
        else:
            return torch.mv(X, self.w_out)


def main():
    u_train, y_train = NARMA(sample_len = 4000)
    u_test, y_test = NARMA(sample_len = 1000)

    esn = ESN()
    esn(u_train, y_train)

    y_predicted = esn(u_test)
    evaluate_prediction(y_predicted, y_test[200:])
    print('NRMSE:', nrmse(y_predicted, y_test[200:]))


if __name__ == '__main__':
    main()
