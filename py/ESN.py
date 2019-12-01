import enum
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from util import spectral_radius as _spectral_radius


class Distribution(enum.Enum):
    gaussian = 1
    uniform = 2
    fixed = 3


class ESN(nn.Module):
    def __init__(self, hidden_nodes=200, spectral_radius=0.9, washout=200,
                 w_in_density=1.0, w_out_density=1.0, w_res_density=1.0,
                 input_scaling=1.0, w_in_distrib=Distribution.uniform,
                 w_res_distrib=Distribution.uniform, awgn_train_std=0.0,
                 awgn_test_std=0.0, adc_quantization=None, readout='pinv',
                 w_ridge=0.00):
        super(ESN, self).__init__()

        self.hidden_nodes = hidden_nodes
        self.spectral_radius = spectral_radius
        self.f = torch.tanh
        self.w_in_density = w_in_density
        self.w_out_density = w_out_density
        self.w_res_density = w_res_density
        self.washout = washout
        self.input_scaling = input_scaling
        self.w_in_distrib = w_in_distrib
        self.w_res_distrib = w_res_distrib
        self.awgn_train_std = awgn_train_std
        self.awgn_test_std = awgn_test_std
        self.adc_quantization = adc_quantization
        self.readout = readout
        self.rr = Ridge(alpha=w_ridge)

        # We can't just mask w_out with the density, as the masked out nodes
        # must be hidden during training as well.
        mask_size = int(self.hidden_nodes*self.w_out_density)
        self.w_out_mask = np.random.choice(self.hidden_nodes, mask_size, replace=False)
        self.w_out_mask = torch.from_numpy(self.w_out_mask)
        self.output_dim = self.w_out_mask.shape[0]

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

        w_out = torch.ones(self.hidden_nodes)

        self.register_buffer('w_res', w_res)
        self.register_buffer('w_in', w_in)
        self.register_buffer('w_out', w_out)


    def forward(self, u, y=None):
        timeseries_len = u.size()[0]
        X = torch.zeros(timeseries_len, self.output_dim)
        x = torch.zeros(self.hidden_nodes)
        v = torch.zeros(timeseries_len)

        for t in range(timeseries_len):
            # Add AWGN to the input signal.
            v_t = torch.zeros(1)
            if y is not None and self.awgn_train_std > 0.0:
                v_t = v_t.normal_(mean=0.0, std=self.awgn_train_std)
            elif y is None and self.awgn_test_std > 0.0:
                v_t = v_t.normal_(mean=0.0, std=self.awgn_test_std)
            v[t] = v_t

            # Calculate the next state of each node as an integration of
            # incoming connections.
            u_t = self.w_in * (u[t] + v_t)
            x_t = self.w_res.mv(x)
            x = self.f(u_t + x_t)
            X[t] = x[self.w_out_mask]

            if self.adc_quantization is not None:
                q = 1/self.adc_quantization
                X[t] = q * torch.round(X[t]/q)

        # Record the previous time series passed through the reservoir.
        self.X = X
        self.v = v

        X = X[self.washout:]
        y = y[self.washout:] if y is not None else y

        if y is not None:
            if self.readout == 'rr':
                self.rr.fit(X, y)
                self.w_out = torch.from_numpy(self.rr.coef_).float()
            elif self.readout == 'pinv':
                self.w_out = torch.mv(torch.pinverse(X), y)
            else:
                raise NotImplementedError('Unknown readout regression method')
        else:
            return torch.mv(X, self.w_out)
