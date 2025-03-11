from torch import nn
import torch

from components.attentionUtils import get_EF
from linearAttentionHead import LinearAttentionHead


class MHAttention(nn.Module):
    """
    Multihead attention
    """

    def __init__(
        self,
        input_size,
        dim,
        channels,
        dim_k,
        nhead,
        dropout,
        checkpoint_level,
        parameter_sharing,
        E_proj,
        F_proj,
        full_attention,
        causal_mask,
        w_o_intermediate_dim=None,
        decoder_mode=False,
        method="learnable",
    ):
        super().__init__()
        self.heads = nn.ModuleList()
        self.input_size = input_size
        self.dim_k = dim_k
        self.channels = channels
        self.causal_mask = causal_mask
        self.checkpoint_level = checkpoint_level
        self.w_o_intermediate_dim = w_o_intermediate_dim

        if parameter_sharing != "layerwise":
            E_proj = get_EF(input_size, dim_k, method, dim)
            F_proj = (
                get_EF(input_size, dim_k, method, dim)
                if parameter_sharing == "none"
                or parameter_sharing == "headwise"
                else E_proj
            )

        self.decoder_mode = decoder_mode
        self.to_q = nn.ModuleList()
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()

        for _ in range(nhead):
            if parameter_sharing == "none":
                E_proj = get_EF(input_size, dim_k, method, dim)
                F_proj = get_EF(input_size, dim_k, method, dim)

            attn = LinearAttentionHead(
                dim, dropout, E_proj, F_proj, causal_mask, full_attention
            )

            self.heads.append(attn)
            self.to_q.append(nn.Linear(channels, dim, bias=False))
            self.to_k.append(nn.Linear(channels, dim, bias=False))
            self.to_v.append(nn.Linear(channels, dim, bias=False))

        if w_o_intermediate_dim is None:
            self.w_o = nn.Linear(dim * nhead, channels)

        else:
            self.w_o_1 = nn.Linear(dim * nhead, w_o_intermediate_dim)
            self.w_o_2 = nn.Linear(w_o_intermediate_dim, channels)
        self.mh_dropout = nn.Dropout(dropout)

    def forward(self, tensor, **kwargs):
        batch_size, input_len, channels = tensor.shape

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
            != (batch_size, input_len, channels)
        ), "Embeddings size must be the same as the input tensor"
        head_outputs = []
        for index, head in enumerate(self.heads):
            Q = self.to_q[index](tensor)
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
            if self.checkpoint_level == "C2":
                head_outputs.append(checkpoint(head, Q, K, V))
            else:
                head_outputs.append(head(Q, K, V, **kwargs))
        out = torch.cat(head_outputs, dim=-1)
        if self.w_o_intermediate_dim is None:
            out = self.w_o(out)
        else:
            out = self.w_o_1(out)
            out = self.w_o_2(out)
        out = self.mh_dropout(out)
        return out
