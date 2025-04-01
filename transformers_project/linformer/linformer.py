import sys
import os

import torch.nn as nn
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from linformer_blocks import LinformerLayers

from torch.utils.checkpoint import checkpoint

from transformer_toolbox import PositionalEmbedding


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
        n_features,
        device,
        d_conversion,
        max_prediction_length,
        dropout_input,
        dropout_multi_head_att,
        dropout_lin_att,
        dim_ff=1024,
        ff_intermediate=None,
        dropout_ff=0.1,
        nhead=4,
        n_layers=2,
        dropout=0.05,
        parameter_sharing="layerwise",
        k_reduce_by_layer=0,
        full_attention=False,
        w_o_intermediate_dim=None,
        method="learnable",
        activation="gelu",
        checkpoint_level="C0",
        causal=False,
    ):
        super().__init__()
        self.input_layer = nn.Linear(n_features, dim)
        self.pos_emb = PositionalEmbedding(dim)
        self.input_dropout = nn.Dropout(dropout_input)

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
            checkpoint_level=checkpoint_level,
            parameter_sharing=parameter_sharing,
            k_reduce_by_layer=k_reduce_by_layer,
            w_o_intermediate_dim=w_o_intermediate_dim,
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

        self.activation = torch.nn.ReLU()

    def forward(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len),
        and all items are ints from [0, num_tokens-1]
        """
        tensor = self.input_layer(tensor)
        tensor = self.pos_emb(tensor).type(tensor.type()) + tensor
        tensor = self.input_dropout(tensor)
        tensor = self.linformer(tensor, **kwargs)

        # Only for time series
        output = tensor.permute(0, 2, 1)

        # Pooling
        output = self.apply_pooling(output)
        output = output.squeeze(-1)

        # Apply Dropout
        output = self.apply_dropout(output)

        output = self.intermediary_layer(output)

        output = self.activation(output)

        output = self.linear_layer(output)

        return output


# class LinformerEncDec(nn.Module):
#     """
#     A complete seq -> seq translation task.
#     Complete with an encoder and a decoder module.
#     """

#     def __init__(
#         self,
#         enc_num_tokens,
#         enc_seq_len,
#         enc_dim,
#         dec_num_tokens,
#         dec_seq_len,
#         dec_dim,
#         enc_dim_k=64,
#         enc_dim_ff=1024,
#         enc_dim_d=None,
#         enc_ff_intermediate=None,
#         dec_ff_intermediate=None,
#         enc_dropout_ff=0.1,
#         enc_nhead=4,
#         enc_depth=2,
#         enc_dropout=0.05,
#         enc_parameter_sharing="layerwise",
#         enc_k_reduce_by_layer=0,
#         enc_full_attention=False,
#         enc_include_ff=True,
#         enc_w_o_intermediate_dim=None,
#         enc_emb_dim=None,
#         enc_method="learnable",
#         dec_dim_k=64,
#         dec_dim_ff=1024,
#         dec_dim_d=None,
#         dec_dropout_ff=0.1,
#         dec_nhead=4,
#         dec_depth=2,
#         dec_dropout=0.05,
#         dec_parameter_sharing="layerwise",
#         dec_k_reduce_by_layer=0,
#         dec_full_attention=False,
#         dec_include_ff=True,
#         dec_w_o_intermediate_dim=None,
#         dec_emb_dim=None,
#         dec_method="learnable",
#         activation="gelu",
#         checkpoint_level="C0",
#     ):
#         super().__init__()
#         self.encoder = LinformerLM(
#             num_tokens=enc_num_tokens,
#             seq_len=enc_seq_len,
#             dim=enc_dim,
#             dim_d=enc_dim_d,
#             dim_ff=enc_dim_ff,
#             dim_k=enc_dim_k,
#             dropout_ff=enc_dropout_ff,
#             nhead=enc_nhead,
#             n_layers=enc_depth,
#             dropout=enc_dropout,
#             parameter_sharing=enc_parameter_sharing,
#             k_reduce_by_layer=enc_k_reduce_by_layer,
#             ff_intermediate=enc_ff_intermediate,
#             full_attention=enc_full_attention,
#             include_ff=enc_include_ff,
#             w_o_intermediate_dim=enc_w_o_intermediate_dim,
#             emb_dim=enc_emb_dim,
#             return_emb=True,
#             activation=activation,
#             checkpoint_level=checkpoint_level,
#             method=enc_method,
#         )
#         self.decoder = LinformerLM(
#             num_tokens=dec_num_tokens,
#             seq_len=dec_seq_len,
#             dim=dec_dim,
#             dim_d=dec_dim_d,
#             dim_ff=dec_dim_ff,
#             dim_k=dec_dim_k,
#             dropout_ff=dec_dropout_ff,
#             nhead=dec_nhead,
#             n_layers=dec_depth,
#             dropout=dec_dropout,
#             ff_intermediate=dec_ff_intermediate,
#             parameter_sharing=dec_parameter_sharing,
#             k_reduce_by_layer=dec_k_reduce_by_layer,
#             method=dec_method,
#             full_attention=dec_full_attention,
#             include_ff=dec_include_ff,
#             w_o_intermediate_dim=dec_w_o_intermediate_dim,
#             emb_dim=dec_emb_dim,
#             decoder_mode=True,
#             causal=True,
#             activation=activation,
#             checkpoint_level=checkpoint_level,
#         )

#     def forward(self, x, y=None, **kwargs):
#         """
#         Input is (batch_size, seq_len),
#         and all items are ints from [0, num_tokens-1]
#         """
#         encoder_output = self.encoder(x, **kwargs)
#         y = y if y is not None else x
#         return self.decoder(y, embeddings=encoder_output)
