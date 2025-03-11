import torch


class LinearAttentionHead(torch.nn.Module):
    """
    Linear attention, as proposed by the linformer paper
    """

    def __init__(
        self, dim, dropout, E_proj, F_proj, causal_mask, full_attention=False
    ):
        super().__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = torch.nn.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention
        self.causal_mask = causal_mask
        self.is_proj_tensor = isinstance(E_proj, torch.Tensor)

    def forward(self, Q, K, V, **kwargs):
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        """
        input_mask = kwargs.get("input_mask")
        embeddings_mask = kwargs.get("embeddings_mask")

        # Instead of classic masking, we have to do this,
        # because the classic mask is of size nxn
        if input_mask is not None:
            # This is for k, v
            mask = input_mask[:, :, None]
            K = K.masked_fill_(~mask, 0.0)
            V = V.masked_fill_(~mask, 0.0)
            del mask

        if embeddings_mask is not None:
            mask = embeddings_mask[:, :, None]
            Q = Q.masked_fill_(~mask, 0.0)
            del mask

        K = K.transpose(1, 2)
        if not self.full_attention:
            if self.is_proj_tensor:
                self.E = self.E.to(K.device)
                K = torch.matmul(K, self.E)
            else:
                K = self.E(K)

        Q = torch.matmul(Q, K)

        P_bar = Q / torch.sqrt(torch.tensor(self.dim).type(Q.type())).to(
            Q.device
        )
        if self.causal_mask is not None:
            self.causal_mask = self.causal_mask.to(Q.device)
            P_bar = P_bar.masked_fill_(~self.causal_mask, float("-inf"))
        P_bar = P_bar.softmax(dim=-1)

        # Only save this when visualizing
        if kwargs["visualize"]:
            self.P_bar = P_bar

        P_bar = self.dropout(P_bar)

        if not self.full_attention:
            V = V.transpose(1, 2)
            if self.is_proj_tensor:
                self.F = self.F.to(V.device)
                V = torch.matmul(V, self.F)
            else:
                V = self.F(V)
            V = V.transpose(1, 2)
        out_tensor = torch.matmul(P_bar, V)

        return out_tensor
