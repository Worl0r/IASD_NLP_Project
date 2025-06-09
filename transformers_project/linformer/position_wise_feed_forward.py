import torch.nn as nn
import os
import sys
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import ActivationFunction, Tensor


class PositionWiseFeedForward(nn.Module):
    """
    The Position-Wise Feed Forward Network sublayer.
    """

    def __init__(
        self,
        d_input: int,
        d_ff: int,
        dropout: float,
        activation: ActivationFunction,
        d_output: int | None = None,
    ) -> None:
        """
        :param d_model: size of vectors throughout the transformer model, i.e.
            input and output sizes for this sublayer
        :param d_ff: an intermediate size
        :param dropout: dropout rate
        :param activation: The activation function for the non-linearity
        """

        super().__init__()

        d_output = d_output if d_output is not None else d_input

        # Linear layers
        self.fc1 = nn.Linear(d_input, d_ff)
        self.fc2 = nn.Linear(d_ff, d_output)

        # Activation
        self.activation = activation()

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Forward prop.

        :param x: input sequences, a tensor of size (N, pad_length, d_model)
        :return: transformed output sequences, a tensor of size (N, pad_length,
            d_model)
        """

        x = self.fc1(x)

        if self.activation:
            x = self.activation(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.dropout2(x)

        return x


class SwiGLU(nn.Module):
    def __init__(self, d_input: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc_in = nn.Linear(d_input, d_ff * 2 // 3 * 2)  # double proj
        self.fc_out = nn.Linear(d_ff * 2 // 3, d_input)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_proj = self.fc_in(x)
        x_val, x_gate = x_proj.chunk(2, dim=-1)
        x = F.silu(x_gate) * x_val
        x = self.dropout(x)
        x = self.fc_out(x)
        return x
