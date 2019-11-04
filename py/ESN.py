import enum
import numpy as np
import torch
import torch.nn as nn

from util import spectral_radius as _spectral_radius


class Distribution(enum.Enum):
    gaussian = 1
    uniform = 2
    fixed = 3


class ESN(nn.Module):
    def __init__(self, hidden_nodes, spectral_radius=0.9, washout=200,
                 w_in_density=1.0, w_out_density=1.0, input_scaling=1.0,
                 w_in_distrib=Distribution.uniform):
        super(ESN, self).__init__()

        self.hidden_nodes = hidden_nodes
        self.spectral_radius = spectral_radius
        self.f = torch.tanh
        self.w_res_density = 0.2
        self.w_in_density = w_in_density
        self.washout = washout
        self.input_scaling = input_scaling
        self.w_in_distrib = w_in_distrib

        # We can't just mask w_out with the density, as the masked out nodes
        # must be hidden during training as well.
        mask_size = int(self.hidden_nodes*w_out_density)
        self.w_out_mask = np.random.choice(self.hidden_nodes, mask_size, replace=False)
        self.w_out_mask = torch.from_numpy(self.w_out_mask)
        self.output_dim = self.w_out_mask.shape[0]

        w_res = torch.rand(self.hidden_nodes, self.hidden_nodes) - 0.5
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

        for t in range(timeseries_len):
            u_t = self.w_in * u[t]
            x_t = self.w_res.mv(x)
            x = self.f(u_t + x_t)
            X[t] = x[self.w_out_mask]

        X = X[self.washout:]
        y = y[self.washout:] if y is not None else y

        if y is not None:
            self.w_out = torch.mv(torch.pinverse(X), y)
        else:
            return torch.mv(X, self.w_out)
