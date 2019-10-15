import torch


def spectral_radius(w):
    return torch.max(torch.abs(torch.eig(w)[0])).item()
