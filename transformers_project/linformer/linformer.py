import sys
import os

import torch.nn as nn
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .linformer_layers import LinformerLayers

from torch.utils.checkpoint import checkpoint

from .positional_encoding import PositionalEmbedding


class LinformerEnc(nn.Module):
    """
    A complete seq -> seq translation task.
    Complete with an encoder and a decoder module.
    """

    def __init__(
        self,
        seq_len,
        dim,
        dim_lin_base,
        vocab_size,
        n_features,
        device,
        d_conversion,
        max_prediction_length,
        dropout_input,
        dropout_multi_head_att,
        dropout_lin_att,
        dim_ff,
        ff_intermediate,
        dropout_ff,
        nhead,
        n_layers,
        dropout,
        parameter_sharing,
        k_reduce_by_layer,
        method,
        activation,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(dim)
        self.input_dropout = nn.Dropout(dropout_input)
        self.normalization = nn.LayerNorm(dim, eps=10e-6)

        self.linformer = LinformerLayers(
            seq_len,
            dim,
            dim_lin_base,
            dim_ff=dim_ff,
            dropout_ff=dropout_ff,
            nhead=nhead,
            n_layers=n_layers,
            dropout_multi_head_att=dropout_multi_head_att,
            dropout_lin_att=dropout_lin_att,
            ff_intermediate=ff_intermediate,
            activation=activation,
            parameter_sharing=parameter_sharing,
            k_reduce_by_layer=k_reduce_by_layer,
            decoder_mode=False,
            method=method,
        )

        # # Pooling
        self.apply_pooling = nn.AdaptiveAvgPool1d(1)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

        # Intermediary layer
        self.intermediary_layer = nn.Linear(dim, d_conversion).to(device)

        # Linear layer
        self.linear_layer = nn.Linear(d_conversion, max_prediction_length).to(
            device
        )

        self.activation = torch.nn.Softmax()

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim
        )

    def forward(self, tensor, *argv, **kwargs):
        """
        Input is (batch_size, seq_len),
        and all items are ints from [0, num_tokens-1]
        """

        tensor = self.embeddings(tensor)

        # We pad if needed
        x = self.pad_sequences(tensor)

        x = self.pos_emb(tensor).type(tensor.type()) + x
        x = self.input_dropout(x)
        x = self.normalization(x)

        # Encoder
        out = self.linformer(x, **kwargs)

        output = self.make_prediction(out)

        return output

    def make_prediction(self, tensor):
        # Only for time series
        output = tensor.permute(0, 2, 1)

        # Pooling
        output = self.apply_pooling(output)
        output = output.squeeze(-1)

        # Apply Dropout
        output = self.apply_dropout(output)

        output = self.intermediary_layer(output)

        # output = self.activation(output)

        output = self.linear_layer(output)

        return output

    def pad_sequences(self, sequences):
        """
        Pad sequences to the same length in a batch.
        """
        batch_size, seq_len, dim = sequences.shape

        if seq_len == self.seq_len:
            return sequences

        if seq_len > self.seq_len:
            return sequences[:, : self.seq_len, :]

        raise ValueError(
            f"Input sequence length {seq_len}"
            f" is less than the expected length {self.seq_len}."
        )
