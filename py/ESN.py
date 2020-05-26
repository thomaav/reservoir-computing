import enum
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from sklearn.linear_model import Ridge

from util import spectral_radius as _spectral_radius
from exception import ESNException
import matrix
import metric


valid_readouts = ['pinv', 'rr']

class Distribution(enum.Enum):
    gaussian = 1
    uniform = 2
    fixed = 3

class ESN(nn.Module):
    def __init__(self, hidden_nodes=200, spectral_radius=0.9, washout=200,
                 w_in_density=1.0, w_res_density=1.0, input_scaling=1.0,
                 w_in_distrib=Distribution.uniform, w_res_distrib=Distribution.uniform,
                 readout='rr', w_ridge=0.00, w_res_type=None, grow_neigh=0,
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
        self.w_ridge = w_ridge
        self.rr = Ridge(alpha=w_ridge, solver='svd')
        self.w_res_type = w_res_type
        self.grow_neigh = grow_neigh

        if self.w_res_type == 'waxman':
            G = matrix.waxman(n=self.hidden_nodes, alpha=1.0, beta=1.0, connectivity='global', **kwargs)
            self.G = G
            A = nx.to_numpy_matrix(G)

            w_res = torch.FloatTensor(A)
            cur_sr = _spectral_radius(w_res)
            self.org_w_res = w_res.clone()
            self.org_spectral_radius = cur_sr
            if self.spectral_radius is None:
                self.spectral_radius = self.org_spectral_radius
            if cur_sr != 0:
                w_res *= self.spectral_radius / cur_sr
        elif self.w_res_type in ['tetragonal', 'hexagonal', 'triangular', 'rectangular']:
            sqrt = np.sqrt(self.hidden_nodes)
            if sqrt - int(sqrt) != 0:
                raise ValueError("Non square number of nodes given for lattice")

            self.sign_frac = None
            if 'sign_frac' in kwargs:
                self.sign_frac = kwargs['sign_frac']
                del kwargs['sign_frac']

            self.dir_frac = None
            if 'dir_frac' in kwargs:
                self.dir_frac = kwargs['dir_frac']
                del kwargs['dir_frac']

            if self.w_res_type == 'tetragonal':
                G = matrix.tetragonal([int(sqrt), int(sqrt)], **kwargs)
            elif self.w_res_type == 'hexagonal':
                G = matrix.hexagonal(int(sqrt) // 2, int(np.ceil(sqrt/2)*2), **kwargs)
            elif self.w_res_type == 'triangular':
                G = matrix.triangular(int(sqrt)+3, int(sqrt)+3, **kwargs)
            elif self.w_res_type == 'rectangular':
                G = matrix.rectangular(int(sqrt), int(sqrt), **kwargs)
            self.G = G

            if self.grow_neigh > 0:
                matrix.grow_neighborhoods(self.G, l=grow_neigh, **kwargs)

            if self.sign_frac is not None:
                matrix.make_weights_negative(self.G, self.sign_frac)

            if self.dir_frac is not None:
                self.G = matrix.make_graph_directed(self.G, self.dir_frac)

            A = nx.to_numpy_matrix(self.G)
            self.hidden_nodes = len(A)
            w_res = torch.FloatTensor(A)

            cur_sr = _spectral_radius(w_res)
            self.org_spectral_radius = cur_sr
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
            if _spectral_radius(w_res) != 0:
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


    def forward(self, u, y=None, u_mc=None, plot=False, kq=False):
        timeseries_len = u.size()[0]
        X = torch.zeros(timeseries_len, self.hidden_nodes)
        x = torch.zeros(self.hidden_nodes)

        for t in range(timeseries_len):
            # Calculate the next state of each node as an integration of
            # incoming connections.
            u_t = self.w_in * u[t]
            x_t = self.w_res.mv(x)
            x = self.f(u_t + x_t)

            X[t] = x

        # Record the previous time series passed through the reservoir.
        self.X = X

        X = X[self.washout:]
        y = y[self.washout:] if y is not None else y

        if y is not None:
            if self.readout == 'rr':
                self.rr.fit(X, y)
            elif self.readout == 'pinv':
                pinv = torch.from_numpy(np.linalg.pinv(X))
                self.w_out = torch.mv(pinv, y)
            else:
                raise ValueError(f'No such readout: {self.readout}')
        elif kq:
            # Ugly way of doing kernel quality without caring about the actually
            # predicted output.
            return
        else:
            if self.readout == 'rr':
                return self.rr.predict(X)
            elif self.readout == 'pinv':
                return torch.mv(X, self.w_out)
            else:
                raise ValueError(f'No such readout: {self.readout}')


    def scale_spectral_radius(self, new_sr):
        cur_sr = _spectral_radius(self.w_res)
        if cur_sr != 0:
            self.w_res *= new_sr / cur_sr


    def memory_capacity(self, washout, u_train, u_test, plot=False):
        if self.readout not in valid_readouts:
            raise ESNException(f'Invalid readout: {self.readout}')

        # To evaluate memory capacity, 1.4*N is suggested as number of output
        # nodes in «Computational analysis of memory capacity in echo state
        # networks».
        output_nodes = int(1.4*self.hidden_nodes)
        washout_len = washout.shape[0]
        train_len = u_train.shape[0]

        self(torch.cat((washout, u_train, u_test), 0), kq=True)
        self.X_train = self.X[washout_len:washout_len+train_len]

        if self.readout == 'rr':
            self.w_outs = [0]*output_nodes
        elif self.readout == 'pinv':
            self.w_outs = torch.zeros(output_nodes, self.hidden_nodes)

        for k in range(1, output_nodes+1):
            if self.readout == 'rr':
                self.w_outs[k-1] = Ridge(alpha=self.w_ridge, solver='svd')
                self.w_outs[k-1].fit(self.X_train[k:, :], u_train[:-k])
            elif self.readout == 'pinv':
                Xplus = torch.pinverse(self.X_train[k:, :])
                self.w_outs[k-1] = torch.mv(Xplus, u_train[:-k])

        self.X_test = self.X[washout_len+train_len:]

        if self.readout == 'rr':
            ys = torch.FloatTensor([rr.predict(self.X_test) for rr in self.w_outs])
        elif self.readout == 'pinv':
            ys = torch.mm(self.w_outs, self.X_test.T)

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


    def remove_hidden_node(self, n):
        if self.w_res_type in ['tetragonal', 'hexagonal', 'triangular']:
            nodes = list(self.G.nodes)
            self.G.remove_node(nodes[n])

            A = nx.to_numpy_matrix(self.G)
            self.hidden_nodes = len(A)

            w_res = torch.FloatTensor(A)
            cur_sr = _spectral_radius(w_res)
            self.org_spectral_radius = cur_sr
            if cur_sr != 0:
                w_res *= self.spectral_radius / cur_sr

            self.w_in = torch.cat([self.w_in[0:n], self.w_in[n+1:]])
            self.w_res = w_res
            self.w_out = torch.cat([self.w_out[0:n], self.w_out[n+1:]])
        elif self.w_res_type is None:
            self.w_in = np.delete(self.w_in, n)
            self.w_out = np.delete(self.w_out, n)
            self.w_res = np.delete(self.w_res, n, axis=1)
            self.w_res = np.delete(self.w_res, n, axis=0)
            self.hidden_nodes = len(self.w_in)
        else:
            raise ESNException(f'Cannot remove hidden node for w_res_type={self.w_res_type}')


    def add_hidden_node(self, node, edges):
        self.G.add_node(node, pos=node)
        for edge in edges:
            self.G.add_edge(edge[0], edge[1])
        self.set_G(self.G)


    def make_edge_undirected(self, edge):
        if self.w_res_type in ['tetragonal', 'hexagonal', 'triangular']:
            u, v = edge[0], edge[1]
            self.G.add_edge(v, u)

            A = nx.to_numpy_matrix(self.G)
            self.hidden_nodes = len(A)

            self.w_res = torch.FloatTensor(A)
            cur_sr = _spectral_radius(self.w_res)
            self.org_spectral_radius = cur_sr
            if cur_sr != 0:
                self.w_res *= self.spectral_radius / cur_sr
        else:
            raise ESNException(f'Cannot remove edge for w_res_type={self.w_res_type}')


    def set_G(self, G):
        if self.w_res_type in ['tetragonal', 'hexagonal', 'triangular']:
            A = nx.to_numpy_matrix(G)
            self.hidden_nodes = len(A)
            self.G = G

            self.w_res = torch.FloatTensor(A)
            cur_sr = _spectral_radius(self.w_res)
            if cur_sr != 0:
                self.w_res *= self.spectral_radius / cur_sr
            else:
                # print('[WARN]: Original spectral radius was 0')
                pass

            self.w_in = torch.ones(self.hidden_nodes)
            self.w_in *= self.input_scaling
            self.w_out = torch.zeros(self.hidden_nodes)
        else:
            raise ESNException(f'Cannot set G for w_res_type={self.w_res_type}')


    def set_readout(self, readout, w_ridge=0.0):
        self.readout = readout
        self.w_ridge = w_ridge
        self.rr = Ridge(alpha=w_ridge, solver='svd')


def find_esn(dataset, required_nrmse, **kwargs):
    attempts = 1000

    for i in range(attempts):
        esn = ESN(**kwargs)
        nrmse = metric.evaluate_esn(dataset, esn)
        if nrmse < required_nrmse:
            return esn

    raise ESNException(f'Could not find suitable ESN within {attempts} attempts')


def create_ESN(G, **kwargs):
    esn = ESN(**kwargs)
    esn.set_G(G)
    return esn


def from_square_G(G):
    params = dict()
    params['hidden_nodes'] = 144
    params['input_scaling'] = 0.1
    params['w_in_distrib'] = Distribution.fixed
    params['w_res_type'] = 'tetragonal'
    params['readout'] = 'rr'
    params['dir_frac'] = 1.0

    esn = ESN(**params)
    esn.set_G(G)
    return esn


def create_delay_line(length):
    lattice = nx.DiGraph()

    for i in range(length):
        node = (0, i)
        lattice.add_node(node, pos=node)
        if i > 0:
            lattice.add_edge(node, (0, i-1))

    return from_square_G(lattice)
