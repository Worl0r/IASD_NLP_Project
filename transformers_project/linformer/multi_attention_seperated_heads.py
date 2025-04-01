import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint

from .projections import get_EF

from .linear_attention_head import LinearAttentionHead


class MHAttention_seperated_heads(nn.Module):
    """
    Multihead attention, with each head being a Linformer Head
    This feeds directly into a feed forward head
    """

    def __init__(
        self,
        seq_len,  # Length of the sequence
        dim,  # Dimension of the model
        dim_lin,  # Dimension of the linear projection
        dim_k,  # Old dimension of the head
        nhead,  # Number of heads
        dropout_multi_head_att,  # Dropout for the multi head attention
        dropout_lin_att,  # Dropout for the linear head attention
        checkpoint_level,
        parameter_sharing,
        E_proj,
        F_proj,
        device,
        w_o_intermediate_dim: int | None = None,
        decoder_mode=False,
        method="learnable",
    ):
        super().__init__()

        # Dimension of the sequence
        self.seq_len = seq_len
        # Dimension of one head
        self.dim_k = dim_k
        # Define the level of the checkpointing
        self.checkpoint_level = checkpoint_level
        # Allow an additional linear layer after the concatenation of the heads
        self.w_o_intermediate_dim = w_o_intermediate_dim

        # If parameter_sharing is not "layerwise",
        # we need to create new projections
        if parameter_sharing != "layerwise":
            E_proj = get_EF(seq_len, dim_lin, dim_k, method)
            if parameter_sharing == "kv":
                F_proj = E_proj
            else:
                # If parameter_sharing is "headwise" or "none",
                F_proj = get_EF(seq_len, dim_lin, dim_k, method)

        self.decoder_mode = decoder_mode

        self.heads = nn.ModuleList()

        self.to_q = nn.ModuleList()
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()

        # Separate the heads
        for _ in range(nhead):
            # Create the linear attention layer
            attn = LinearAttentionHead(
                dim, dropout_lin_att, E_proj, F_proj, device
            )
            self.heads.append(attn)

            # Tranform the dimension of the input into dimension of the head
            self.to_q.append(nn.Linear(dim_k, dim, bias=False))
            self.to_k.append(nn.Linear(dim_k, dim, bias=False))
            self.to_v.append(nn.Linear(dim_k, dim, bias=False))

        if w_o_intermediate_dim is None:
            # Come back to the original dimension with one linear layer
            self.w_o = nn.Linear(dim_k * nhead, dim)
        else:
            # Come back to the ordinal dimension with two linear layers
            self.w_o_1 = nn.Linear(dim * nhead, w_o_intermediate_dim)
            self.w_o_2 = nn.Linear(w_o_intermediate_dim, dim)

        self.mh_dropout = nn.Dropout(dropout_multi_head_att)

    def forward(self, tensor, **kwargs):
        batch_size, seq_len, dim = tensor.shape

        assert not (self.decoder_mode and "embeddings" not in kwargs), (
            "Embeddings must be supplied if decoding"
        )
        assert not (
            "embeddings" in kwargs
            and (
                kwargs["embeddings"].shape[0],
                kwargs["embeddings"].shape[1],
                kwargs["embeddings"].shape[2],
            )
            != (batch_size, seq_len, dim)
        ), "Embeddings size must be the same as the input tensor"

        head_outputs = []

        for index, head in enumerate(self.heads):
            # Basic projections
            Q = self.to_q[index](tensor)

            # We take the encoded tensor if needed (decoder)
            K = (
                self.to_k[index](tensor)
                if not self.decoder_mode
                else self.to_k[index](kwargs["embeddings"])
            )
            V = (
                self.to_v[index](tensor)
                if not self.decoder_mode
                else self.to_v[index](kwargs["embeddings"])
            )

            # Checkpointing to save memory
            if self.checkpoint_level == "MHAttention":
                head_outputs.append(
                    checkpoint(head, Q, K, V, use_reentrant=True)
                )
            else:
                head_outputs.append(head(Q, K, V, **kwargs))

        # Concatenation
        out = torch.cat(head_outputs, dim=-1)

        # We come back to the original dimension with one or two linear layers
        if self.w_o_intermediate_dim is None:
            out = self.w_o(out)
        else:
            out = self.w_o_1(out)
            out = self.w_o_2(out)

        # Dropout
        out = self.mh_dropout(out)

        return out
