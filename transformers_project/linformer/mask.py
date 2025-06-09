import torch


def gen_causal_mask(seq_len, dim_k):
    """
    Generates a causal mask of size (seq_len, dim_k) for linformer
    Else, it generates (seq_len, seq_len) for full attention
    """

    return (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
