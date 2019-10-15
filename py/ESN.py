import torch
import torch.nn as nn

from util import spectral_radius as _spectral_radius


class ESN(nn.Module):
    def __init__(self, hidden_nodes, spectral_radius=0.9, washout=200,
                 w_in_sparsity=1.0, w_out_sparsity=1.0, input_scaling=1.0):
        super(ESN, self).__init__()

        self.hidden_nodes = hidden_nodes
        self.spectral_radius = spectral_radius
        self.f = torch.tanh
        self.w_res_sparsity = 0.2
        self.w_in_sparsity = w_in_sparsity
        self.washout = washout
        self.input_scaling = input_scaling

        # We can't just mask w_out with the sparsity, as the masked out nodes
        # must be hidden during training as well.
        self.w_out_sparsity = w_out_sparsity
        self.w_out_mask = torch.ones(self.hidden_nodes)
        self.w_out_mask[torch.rand(self.hidden_nodes) > self.w_out_sparsity] = 0.0

        w_res = torch.rand(self.hidden_nodes, self.hidden_nodes) - 0.5
        w_res[torch.rand(self.hidden_nodes, self.hidden_nodes) > self.w_res_sparsity] = 0.0
        w_res *= self.spectral_radius / _spectral_radius(w_res)
        w_in = (torch.rand(self.hidden_nodes) - 0.5)
        w_in[torch.rand(self.hidden_nodes) > self.w_in_sparsity] = 0.0
        w_in *= self.input_scaling
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
            X[t] = x * self.w_out_mask

        X = X[self.washout:]
        y = y[self.washout:] if y is not None else y

        if y is not None:
            self.w_out = torch.mv(torch.pinverse(X), y)
        else:
            return torch.mv(X, self.w_out)
