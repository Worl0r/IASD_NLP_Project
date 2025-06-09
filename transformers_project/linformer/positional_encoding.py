import math

import torch
import torch.nn as nn
from torch.nn.init import constant_

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int) -> None:
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)

        position = torch.arange(
            0, max_seq_length, dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        output = torch.zeros(x.size(0), x.size(1), self.pe.size(-1))

        for i in range(x.size(-1)):
            output = x[:, :, i].unsqueeze(-1) + self.pe[:, : x.size(1)].to(
                x.device
            )

        return output


class PositionalEmbedding(nn.Module):
    """
    Standard positional embedding.
    From the paper "Attention is all you need".
    Changed the constant from 10k to 100k,
    since this may be better for longer sequence lengths.
    """

    def __init__(self, channels: int, constant: int = 100000) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            constant ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        pos = torch.arange(tensor.shape[1], device=tensor.device).type(
            self.inv_freq.type()
        )
        sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return emb[None, :, :]
