import torch.nn as nn


class Residual(nn.Module):
    """
    Implemenation taken from
    https://github.com/lucidrains/sinkhorn-transformer/blob/master/sinkhorn_transformer/sinkhorn_transformer.py
    However, I do postnorm instead of prenorm.
    """

    def __init__(self, fn, input_channels, output_channels):
        super().__init__()

        self.fn = fn
        self.resample = (
            nn.Linear(input_channels, output_channels)
            if input_channels != output_channels
            else None
        )
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, tensor, **kwargs):
        if self.resample is not None:
            tensor = self.resample(tensor) + self.fn(tensor, **kwargs)
        else:
            tensor = tensor + self.fn(tensor, **kwargs)

        tensor = self.norm(tensor)
        return tensor
