from torch import nn
import torch

from utils.types import ActivationFunction, Tensor
from components.positionalEncoding import PositionalEncoding


class Transformers(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        max_prediction_length: int,
        d_model: int,
        num_heads: int,
        n_layers: int,
        d_conversion: int,
        d_ff: int,
        nbr_classes: int,
        dropout: float,
        device: torch.device,
        eps: float,
        use_positional_encoding: bool,
        activationEncoderLayers: ActivationFunction = nn.ReLU,
    ) -> None:
        super().__init__()
        self.use_positional_encoding = use_positional_encoding

        self.embedding = nn.Embedding(d_model)

        self.positional_encoding = PositionalEncoding(
            max_seq_length=src_vocab_size, d_model=d_model
        )

        self.device = device

        # Inpout layer
        self.src_input_layer = nn.Linear(nbr_classes, d_model)

        # we remove the demand count classes because we want to predict it
        self.tgt_input_layer = nn.Linear(nbr_classes - 1, d_model)

        self.input_normalization = nn.LayerNorm(d_model, eps=eps)

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activationEncoderLayers,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )

        # Decoder Layers
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activationEncoderLayers,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )

        # Pooling
        self.apply_pooling = nn.AdaptiveAvgPool1d(1)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

        # Intermediary layer
        self.intermediary_layer = nn.Linear(d_model, d_conversion).to(device)

        # Linear layer
        self.linear_layer = nn.Linear(d_model, max_prediction_length).to(
            device
        )

        # Activation function
        self.activation = nn.ReLU()

        # Initialize weights
        self.init_weights()

    def generate_mask(self, tgt, label="known"):
        """
        Generates only the causal mask for multi-head attention in the target.

        Arguments:
        - tgt : target tensor with shape (batch_size, tgt_seq_len, num_features)

        Returns:
        - tgt_mask : causal mask for the target tensor with shape (tgt_seq_len, tgt_seq_len)
        """

        # Target sequence length
        tgt_seq_len = tgt.size(1)

        if label == "known":
            tgt_mask = torch.ones(tgt_seq_len, tgt_seq_len).bool()
            tgt_mask = tgt_mask.to(tgt.device)
            return tgt_mask

        elif label == "unknown":
            # Causal mask for the target
            # Prevents each position from attending to future positions
            # Shape: (tgt_seq_len, tgt_seq_len)
            tgt_mask = torch.triu(
                torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1
            ).bool()
            tgt_mask = tgt_mask.to(
                tgt.device
            )  # Ensures the mask is on the same device as `tgt`

            # Dimensions: (tgt_seq_len, tgt_seq_len)
            return tgt_mask
        else:
            raise ValueError(
                "The label should be either 'known' or 'unknown'."
            )

    def init_weights(self) -> None:
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        tgt_mask = self.generate_mask(tgt, "known")

        if self.use_positional_encoding:
            return
            # src_embedded = self.apply_dropout(
            #     self.embedding(src) + self.positional_encoding(src)
            # )
            # tgt_embedded = self.apply_dropout(
            #     self.embedding(tgt) + self.positional_encoding(tgt)
            # )
        else:
            src_embedded = self.src_input_layer(src).to(self.device)
            src_embedded = self.input_normalization(src_embedded).to(
                self.device
            )

            tgt_embedded = self.tgt_input_layer(tgt).to(self.device)
            tgt_embedded = self.input_normalization(tgt_embedded).to(
                self.device
            )

        enc_output = src_embedded

        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(x=enc_output, mask=None)

        dec_output = tgt_embedded
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer(dec_output, enc_output, tgt_mask)

        output = dec_output.permute(0, 2, 1)

        # Pooling
        output = self.apply_pooling(output)
        output = output.squeeze(-1)

        # Apply Dropout
        output = self.apply_dropout(output)

        output = self.intermediary_layer(output)

        # Reduce the sequence dimension (e.g., by taking the mean)
        # output = dec_output.mean(dim=1)

        output = self.activation(output)

        output = self.linear_layer(output)

        return output
