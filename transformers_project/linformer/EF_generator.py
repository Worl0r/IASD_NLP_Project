import torch
import torch.nn as nn


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
