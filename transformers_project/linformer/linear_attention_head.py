import torch
import torch.nn as nn


class LinearAttentionHead(nn.Module):
    """
    Linear attention, as proposed by the linformer paper
    """

    def __init__(
        self,
        dim,
        dropout_lin_att,
        E_proj,
        F_proj,
        device,
        causal_mask,
    ):
        super().__init__()

        # Linear layers
        self.E = E_proj.to(device)
        self.F = F_proj.to(device)

        # Dimension of the model (embeddings)
        self.dim = dim
        self.dropout = nn.Dropout(dropout_lin_att)
        self.device = device

        self.is_proj_tensor = isinstance(E_proj, torch.Tensor)

        self.causal_mask = causal_mask

    def forward(self, Q, K, V, **kwargs):
        # K shape: (bt, seq, dim_k)
        # E (E.T in the paper) shape: (seq, dim_lin)
        # F (F.T in the paper) shape: (seq, dim_lin)
        # Q shape: (bt, seq, dim_k)
        # V shape: (bt, seq, dim_k)

        K = K.to(self.device)
        Q = Q.to(self.device)
        V = V.to(self.device)

        ## Retrive potential mask (if decoder)
        input_mask = kwargs.get("input_mask")
        encoded_sequences = kwargs.get("embedding_mask")

        # Mask if needed
        K, V = self.mask_input(input_mask, K, V)
        Q = self.mask_embeddings(encoded_sequences, Q)

        # Transpose
        K = K.permute(0, 2, 1)  # (bt, dim_k, seq)

        # Linear projection of the key with E
        K_proj = torch.matmul(K, self.E) if self.is_proj_tensor else self.E(K)
        # (in the paper it is (E_i.K.W_i^K).T)
        # K_proj shape: (bt, dim_k, dim_lin)

        # Similarity between Q and K_proj
        Q = torch.matmul(Q, K_proj)
        # new Q shape: (bt, seq, dim_lin)

        # Standardization form Attention is All you Need
        simQK_proj = Q / torch.sqrt(torch.tensor(self.dim).type(Q.type())).to(
            self.device
        )

        # Only in decoder mode we add causal mask
        # DEBUG
        # if self.causal_mask is not None:
        #     self.causal_mask = self.causal_mask.to(self.device)
        #     simQK_proj = simQK_proj.masked_fill_(
        #         ~self.causal_mask, float("-inf")
        #     )

        # Softmax function
        P_bar = simQK_proj.softmax(dim=-1)
        # P_bar shape egual to new Q

        # Dropout: Choice
        P_bar = self.dropout(P_bar)

        # Transpose
        V = V.permute(0, 2, 1)
        # V shape : (bt, dim_k, seq)

        # Linear projection of the value with F
        V_proj = torch.matmul(V, self.F) if self.is_proj_tensor else self.F(V)
        # (in the paper it is (F_i.V.W_i^V).T)
        # V_proj shape : (bt, dim_k, dim_lin)

        # Transpose
        V_proj_T = V_proj.permute(0, 2, 1)
        # V_proj_T shape: (bt, dim_lin, dim_k)

        P_bar = torch.matmul(P_bar, V_proj_T)
        # (in the paper it is (P_bar_i.V_proj_i).T)
        # P_bar shape : (bt, seq, dim_k)

        return P_bar

    def mask_input(self, input_mask, K, V):
        if input_mask is not None:
            # This is for k, v
            mask = input_mask[:, :, None]
            K = K.masked_fill_(~mask, 0.0)
            V = V.masked_fill_(~mask, 0.0)
            del mask

        return K, V

    def mask_embeddings(self, embeddings_mask, Q):
        # If we use the decoder, we need to mask the Q
        if embeddings_mask is not None:
            mask = embeddings_mask[:, :, None]
            Q = Q.masked_fill_(~mask, 0.0)
            del mask

        return Q
