import sys
import os

import torch.nn as nn

from .projections import (
    get_EF,
    get_act,
)
from .multi_attention_seperated_heads import MHAttention_seperated_heads


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformer_toolbox import (
    PositionWiseFeedForward,
)

from .residual import Residual


class LinformerLayers(nn.Module):
    """
    My attempt at reproducing the Linformer Paper
    https://arxiv.org/pdf/2006.04768.pdf
    """

    def __init__(
        self,
        seq_len,
        dim,
        dim_lin_base,
        dim_ff=256,
        dropout_ff=0.15,
        nhead=4,
        n_layers=1,
        dropout_multi_head_att=0.1,
        dropout_lin_att=0.1,
        activation="gelu",
        checkpoint_level="MHAttention",
        parameter_sharing="layerwise",
        k_reduce_by_layer=0,
        w_o_intermediate_dim=None,
        decoder_mode=False,
        method="learnable",
        ff_intermediate=None,
    ):
        super().__init__()

        self.activation = get_act(activation)

        assert (
            checkpoint_level == "C0"
            or checkpoint_level == "Linformer"
            or checkpoint_level == "MHAttention"
        ), "Checkpoint level has to be either C0, Linformer, or MHAttention."
        assert (
            parameter_sharing == "none"
            or parameter_sharing == "headwise"
            or parameter_sharing == "kv"
            or parameter_sharing == "layerwise"
        ), (
            "The `parameter_sharing` flag has to be either"
            " 'none', 'headwise', 'kv', or 'layerwise'."
        )
        assert dim % nhead == 0, "`d_model` must be divisible by `nhead`!"

        assert not (ff_intermediate and parameter_sharing == "layerwise"), (
            "Parameter sharing must not be"
            " layerwise if ff_intermediate is enabled!"
        )
        assert not (ff_intermediate and decoder_mode), (
            "Raising the dimension in the middle cannot be"
            " done in the decoder!"
        )

        layers = nn.ModuleList()
        self.decoder_mode = decoder_mode
        self.seq_len = seq_len
        self.dim = dim
        self.checkpoint_level = checkpoint_level
        self.n_layers = n_layers
        self.nhead = nhead

        # Define the dimension for the heads
        dim_k = dim // nhead

        # If parameter_sharing is not "layerwise",
        if parameter_sharing == "layerwise":
            E_proj = get_EF(seq_len, dim, dim_k, method)
        else:
            E_proj = None

        def get_attn(input_dim, curr_dim_lin):
            # curr_dim_lin is the current dim_lin
            return MHAttention_seperated_heads(
                seq_len,
                input_dim,
                curr_dim_lin,
                dim_k,
                nhead,
                dropout_multi_head_att,
                dropout_lin_att,
                checkpoint_level,
                parameter_sharing,
                E_proj,
                E_proj,
                w_o_intermediate_dim,
                decoder_mode=False,
                method=method,
            )

        def get_attn_context(attn_dim, curr_dim_k):
            return MHAttention_seperated_heads(
                seq_len,
                dim,
                attn_dim,
                curr_dim_k,
                nhead,
                dropout_multi_head_att,
                dropout_lin_att,
                checkpoint_level,
                parameter_sharing,
                E_proj,
                E_proj,
                w_o_intermediate_dim,
                decoder_mode=True,
                method=method,
            )

        def get_ff(input_dim, output_dim, activation_function):
            return PositionWiseFeedForward(
                d_input=input_dim,
                d_ouput=output_dim,
                d_ff=dim_ff,
                dropout=dropout_ff,
                activation=activation_function,
            )

        for index in range(n_layers):
            # If we want to use ffn with intermediate dimension and
            # we are not in the decoder, not int the fist layer
            # and not the last layer
            input_dim = (
                ff_intermediate
                if (index != 0 and ff_intermediate is not None)
                and not decoder_mode
                else dim
            )
            output_dim = (
                ff_intermediate
                if (index != n_layers - 1 and ff_intermediate is not None)
                and not decoder_mode
                else dim
            )

            # Encoder part: There is the option to decrease
            attn_layer = get_attn(
                input_dim, max(1, dim_lin_base - index * k_reduce_by_layer)
            )
            ff_layer = get_ff(input_dim, output_dim, self.activation)

            attn_layer = Residual(attn_layer, input_dim, input_dim)

            ff_layer = Residual(ff_layer, input_dim, output_dim)

            layers.extend([attn_layer, ff_layer])

            # Decoder part
            if not self.decoder_mode:
                continue

            attn_context = get_attn_context(
                dim, max(1, dim_lin_base - index * k_reduce_by_layer)
            )

            ff_context = get_ff(dim, dim, self.activation)

            attn_context = Residual(attn_context, dim, dim)

            ff_context = Residual(ff_context, dim, dim)

            layers.extend([attn_context, ff_context])

        self.seq = layers

    def forward(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len, dim)
        """
        _, n, d = tensor.shape

        assert n == self.seq_len, (
            "This tensor is of the wrong size. Dimension 1 has to match"
            " the `seq_len` flag"
        )
        assert d == self.dim, (
            "This tensor is of the wrong size. Dimension 2 has to match the"
            " `dim` flag"
        )
        assert self.checkpoint_level == "C0" if kwargs else True, (
            "Cannot run checkpointing when using kwargs."
            " Please set the checkpoint level to `C0`"
        )
        assert "embeddings" not in kwargs or self.decoder_mode, (
            "If decoding, needs to be initialized with `decoder_mode=True`"
        )

        for layer in self.seq:
            if self.checkpoint_level != "Linformer":
                tensor = checkpoint(layer, tensor, use_reentrant=True)
            else:
                tensor = layer(tensor, **kwargs)
        return tensor
