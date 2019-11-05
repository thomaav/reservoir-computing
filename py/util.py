import torch
import math


def spectral_radius(w):
    return torch.max(torch.abs(torch.eig(w)[0])).item()


def snr(var_u, var_v):
    return 10*math.log10(var_u/var_v)


def v_std_from_snr(var_u, snr):
    return math.sqrt(var_u / (10**(snr/10)))
