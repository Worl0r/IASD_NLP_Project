import torch
from torch import nn


def get_EF(input_size, dim, method="learnable", head_dim=None, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for
    no additional params.
    """
    assert (
        method == "learnable"
        or method == "convolution"
        or method == "no_params"
    ), (
        "The method flag needs to be either 'learnable', 'convolution',"
        " or 'no_params'!"
    )

    if method == "convolution":
        conv = nn.Conv1d(
            head_dim,
            head_dim,
            kernel_size=int(input_size / dim),
            stride=int(input_size / dim),
        )
        return conv

    if method == "no_params":
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1 / dim)
        return mat

    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin


class FeedForward(nn.Module):
    """
    Standard Feed Forward Layer
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        ff_dim,
        dropout,
        activation="gelu",
    ):
        super().__init__()
        self.w_1 = nn.Linear(input_channels, ff_dim)
        self.w_2 = nn.Linear(ff_dim, output_channels)
        self.activation = get_act(activation)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tensor, **kwargs):
        tensor = self.w_1(tensor)
        tensor = self.activation(tensor)
        tensor = self.dropout(tensor)
        tensor = self.w_2(tensor)
        tensor = self.dropout2(tensor)
        return tensor


class Residual(nn.Module):
    """
    Implemenation taken from
    https://github.com/lucidrains/sinkhorn-transformer/blob/master/sinkhorn_transformer/sinkhorn_transformer.py
    However, I do postnorm instead of prenorm.
    """

    def __init__(self, fn, input_channels=0, output_channels=0):
        super().__init__()
        self.fn = fn
        self.resample = (
            nn.Linear(input_channels, output_channels)
            if input_channels != output_channels
            else None
        )
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, tensor, **kwargs):
        if self.resample is not None:
            tensor = self.resample(tensor) + self.fn(tensor, **kwargs)
            tensor = self.norm(tensor)
            return tensor
        tensor = tensor + self.fn(tensor, **kwargs)
        tensor = self.norm(tensor)
        return tensor
