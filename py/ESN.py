import enum
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from sklearn.linear_model import Ridge

from util import spectral_radius as _spectral_radius
import matrix


class Distribution(enum.Enum):
    gaussian = 1
    uniform = 2
    fixed = 3


class ESN(nn.Module):
    def __init__(self, hidden_nodes=200, spectral_radius=0.9, washout=200,
                 w_in_density=1.0, w_res_density=1.0, input_scaling=1.0,
                 w_in_distrib=Distribution.uniform, w_res_distrib=Distribution.uniform,
                 readout='pinv', w_ridge=0.00, w_res_type=None, grow_neigh=0,
                 **kwargs):
        super().__init__()

        self.hidden_nodes = hidden_nodes
        self.spectral_radius = spectral_radius
        self.f = torch.tanh
        self.w_in_density = w_in_density
        self.w_res_density = w_res_density
        self.washout = washout
        self.input_scaling = input_scaling
        self.w_in_distrib = w_in_distrib
        self.w_res_distrib = w_res_distrib
        self.readout = readout
        self.rr = Ridge(alpha=w_ridge)
        self.w_res_type = w_res_type
        self.grow_neigh = grow_neigh

        if self.w_res_type == 'waxman':
            G = matrix.waxman(n=self.hidden_nodes, alpha=1.0, beta=1.0, connectivity='global', **kwargs)
            self.G = G
            A = nx.to_numpy_matrix(G)
            w_res = torch.FloatTensor(A)
            w_res *= self.spectral_radius / _spectral_radius(w_res)
        elif self.w_res_type in ['tetragonal', 'hexagonal', 'triangular', 'rectangular']:
            sqrt = np.sqrt(self.hidden_nodes)
            if sqrt - int(sqrt) != 0:
                raise ValueError("Non square number of nodes given for lattice")

            sign_frac = None
            if 'sign_frac' in kwargs:
                sign_frac = kwargs['sign_frac']
                del kwargs['sign_frac']

            dir_frac = None
            if 'dir_frac' in kwargs:
                dir_frac = kwargs['dir_frac']
                del kwargs['dir_frac']

            if self.w_res_type == 'tetragonal':
                G = matrix.tetragonal([int(sqrt), int(sqrt)], **kwargs)
            elif self.w_res_type == 'hexagonal':
                G = matrix.hexagonal(int(sqrt) // 2, int(np.ceil(sqrt/2)*2), **kwargs)
            elif self.w_res_type == 'triangular':
                G = matrix.triangular(int(sqrt) * 2, int(sqrt), **kwargs)
            elif self.w_res_type == 'rectangular':
                G = matrix.rectangular(int(sqrt), int(sqrt), **kwargs)
            self.G = G

            if self.grow_neigh > 0:
                matrix.grow_neighborhoods(self.G, l=grow_neigh, **kwargs)

            if sign_frac is not None:
                matrix.make_weights_negative(self.G, sign_frac)

            if dir_frac is not None:
                self.G = matrix.make_graph_directed(self.G, dir_frac)

            A = nx.to_numpy_matrix(self.G)
            self.hidden_nodes = len(A)
            w_res = torch.FloatTensor(A)
            cur_sr = _spectral_radius(w_res)
            if cur_sr != 0:
                w_res *= self.spectral_radius / cur_sr
        else:
            if self.w_res_distrib == Distribution.gaussian:
                w_res = torch.empty(self.hidden_nodes, hidden_nodes).normal_(mean=0.0, std=1.0)
            elif self.w_res_distrib == Distribution.uniform:
                w_res = torch.rand(self.hidden_nodes, self.hidden_nodes) - 0.5
            elif self.w_res_distrib == Distribution.fixed:
                w_res = torch.ones(self.hidden_nodes, self.hidden_nodes)

            w_res[torch.rand(self.hidden_nodes, self.hidden_nodes) > self.w_res_density] = 0.0
            w_res *= self.spectral_radius / _spectral_radius(w_res)

        if self.w_in_distrib == Distribution.gaussian:
            w_in = torch.empty(self.hidden_nodes).normal_(mean=0.0, std=1.0)
        elif self.w_in_distrib == Distribution.uniform:
            w_in = torch.rand(self.hidden_nodes) - 0.5
        elif self.w_in_distrib == Distribution.fixed:
            w_in = torch.ones(self.hidden_nodes)

        w_in[torch.rand(self.hidden_nodes) > self.w_in_density] = 0.0
        w_in *= self.input_scaling

        w_out = torch.zeros(self.hidden_nodes)

        self.register_buffer('w_res', w_res)
        self.register_buffer('w_in', w_in)
        self.register_buffer('w_out', w_out)


    def forward(self, u, y=None, u_mc=None, plot=False):
        timeseries_len = u.size()[0]
        X = torch.zeros(timeseries_len, self.hidden_nodes)
        x = torch.zeros(self.hidden_nodes)
        v = torch.zeros(timeseries_len)

        for t in range(timeseries_len):
            # Calculate the next state of each node as an integration of
            # incoming connections.
            u_t = self.w_in * u[t]
            x_t = self.w_res.mv(x)
            x = self.f(u_t + x_t)

            X[t] = x

        # Record the previous time series passed through the reservoir.
        self.X = X

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(self.X[self.noisy_mask])
            plt.show()

        X = X[self.washout:]
        y = y[self.washout:] if y is not None else y

        if y is not None:
            if self.readout == 'rr':
                self.rr.fit(X, y)
                self.w_out = torch.from_numpy(self.rr.coef_).float()
            elif self.readout == 'pinv':
                pinv = torch.from_numpy(np.linalg.pinv(X))
                self.w_out = torch.mv(pinv, y)
            else:
                raise ValueError(f'No such readout: {self.readout}')
        else:
            if self.readout == 'rr':
                return self.rr.predict(X)
            elif self.readout == 'pinv':
                return torch.mv(X, self.w_out)
            else:
                raise ValueError(f'No such readout: {self.readout}')


    def memory_capacity(self, washout, u_train, u_test, plot=False):
        # To evaluate memory capacity, 1.4*N is suggested as number of output
        # nodes in «Computational analysis of memory capacity in echo state
        # networks».
        output_nodes = int(1.4*self.hidden_nodes)
        washout_len = washout.shape[0]
        train_len = u_train.shape[0]

        self(torch.cat((washout, u_train, u_test), 0))
        self.X_train = self.X[washout_len:washout_len+train_len]

        self.w_outs = torch.zeros(output_nodes, self.hidden_nodes)
        if self.readout == 'pinv':
            Xplus = torch.pinverse(self.X[washout_len:washout_len+train_len])
        for k in range(1, output_nodes+1):
            if self.readout == 'rr':
                X = self.X[washout_len:washout_len+train_len]
                self.rr.fit(X[k:, :], u_train[:-k])
                self.w_outs[k-1] = torch.from_numpy(self.rr.coef_).float()
            elif self.readout == 'pinv':
                self.w_outs[k-1] = torch.mv(Xplus[:, k:], u_train[:-k])

        X_test = self.X[washout_len+train_len:]
        ys = torch.mm(self.w_outs, X_test.T)

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(u_test)
            for k in range(output_nodes):
                plt.plot(ys[k][k+1:])
            plt.show()

        mc = 0
        self.mcs = []
        for k in range(output_nodes):
            u_tk = u_test[:-(k+1)]
            numerator = (np.cov(u_tk, ys[k][k+1:], bias=True)[0][1])**2
            denominator = torch.var(u_tk)*torch.var(ys[k][k+1:])
            _mc = numerator/denominator
            mc += _mc
            self.mcs.append(_mc)
        return float(mc)
