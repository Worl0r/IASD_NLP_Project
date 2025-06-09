import torch


def identity(x):
    return x


def get_act(activation):
    if activation == "gelu":
        return torch.nn.GELU
    if activation == "relu":
        return torch.nn.ReLU
    if activation == "swish":
        return torch.nn.SiLU
    if activation == "swiglu":
        return activation
    else:
        raise ValueError("Activation not found!")
