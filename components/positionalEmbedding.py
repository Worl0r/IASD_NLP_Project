import torch


class PositionalEmbedding(torch.nn.Module):
    """
    Standard positional embedding.
    From the paper "Attention is all you need".
    Changed the constant from 10k to 100k, since this may be better for longer
    sequence lengths.
    """

    def __init__(self, channels, constant=10e4):
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
