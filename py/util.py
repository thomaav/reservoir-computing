import torch
import math


def spectral_radius(w):
    return torch.max(torch.abs(torch.eig(w)[0])).item()


def snr(var_u, var_v):
    return 10*math.log10(var_u/var_v)
