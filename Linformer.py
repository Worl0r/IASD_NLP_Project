import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def identity(x, *args, **kwargs):
    return x


def get_act(activation):
    if activation == "gelu":
        return F.gelu
    if activation == "relu":
        return F.relu
    if activation == "swish":
        return Swish

    return None


class PositionalEmbedding(nn.Module):
    """
    Standard positional embedding.
    From the paper "Attention is all you need".
    Changed the constant from 10k to 100k, since this may be better for longer
    sequence lengths.
    """

    def __init__(self, channels):
        super().__init__()
        inv_freq = 1.0 / (
            100000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        pos = torch.arange(tensor.shape[1], device=tensor.device).type(
            self.inv_freq.type()
        )
        sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return emb[None, :, :]


class FeedForward(nn.Module):
    """
    Standard Feed Forward Layer
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        ff_dim,
        dropout,
        activation="gelu",
    ):
        super().__init__()
        self.w_1 = nn.Linear(input_channels, ff_dim)
        self.w_2 = nn.Linear(ff_dim, output_channels)
        self.activation = get_act(activation)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tensor, **kwargs):
        tensor = self.w_1(tensor)
        tensor = self.activation(tensor)
        tensor = self.dropout(tensor)
        tensor = self.w_2(tensor)
        tensor = self.dropout2(tensor)
        return tensor


def get_EF(input_size, dim, method="learnable", head_dim=None, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for
    no additional params.
    """
    assert (
        method == "learnable"
        or method == "convolution"
        or method == "no_params"
    ), (
        "The method flag needs to be either 'learnable', 'convolution',"
        " or 'no_params'!"
    )

    if method == "convolution":
        conv = nn.Conv1d(
            head_dim,
            head_dim,
            kernel_size=int(input_size / dim),
            stride=int(input_size / dim),
        )
        return conv

    if method == "no_params":
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1 / dim)
        return mat

    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin


class LinearAttentionHead(nn.Module):
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
        self.dropout = nn.Dropout(dropout)
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
