import torch
import torch.functional as F


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def identity(x, *args, **kwargs):
    return x


def get_act(activation):
    if activation == "gelu":
        return F.gelu
    if activation == "relu":
        return F.relu
    if activation == "swish":
        return Swish

    return None
