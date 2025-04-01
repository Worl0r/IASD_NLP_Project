import torch
import torch.nn as nn


def identity(x):
    return x


def get_act(activation):
    if activation == "gelu":
        return torch.nn.GELU
    if activation == "relu":
        return torch.nn.ReLU
    if activation == "swish":
        return torch.nn.SiLU
    else:
        raise ValueError("Activation not found!")


def gen_causal_mask(seq_len, dim_k, full_attention=False):
    """
    Generates a causal mask of size (seq_len, dim_k) for linformer
    Else, it generates (seq_len, seq_len) for full attention
    """
    if full_attention:
        return (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    return (torch.triu(torch.ones(dim_k, seq_len)) == 1).transpose(0, 1)


def get_EF(seq_len, dim_lin, dim_k=None, method="learnable", bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    """

    assert not (method == "convolution" and dim_k is None), (
        "If you want to use convolution, you need to specify the dim_k"
        " parameter!"
    )

    if method == "convolution":
        conv = nn.Conv1d(
            dim_k,
            dim_k,
            kernel_size=int(seq_len / dim_lin),
            stride=int(seq_len / dim_lin),
        )
        return conv

    elif method == "no_params":
        mat = torch.zeros((seq_len, dim_lin))
        torch.nn.init.normal_(mat, mean=0.0, std=1 / dim_lin)
        return mat

    elif method == "learnable":
        # Linear layer
        lin = nn.Linear(seq_len, dim_lin, bias)
        torch.nn.init.xavier_normal_(lin.weight)
        return lin

    else:
        raise ValueError(
            "The method flag needs to be either 'learnable',"
            " 'convolution', or 'no_params'!"
        )


def define_dim_k(dim_d, error):
    return dim_d / error**2
