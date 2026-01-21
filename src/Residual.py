import torch

def Residual_potential(descriptors: torch.Tensor) -> torch.Tensor:
    """get residual potential
    descriptors: (Npairs,Nradial+Nangular) tensor of charge weighteddescriptors
    return: (float) residual potential energy
    """
    return None
