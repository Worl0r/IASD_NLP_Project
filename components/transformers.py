import torch


class Linformer(torch.nn.Module):
    def __init__(
        self,
        input_size,
        channels,
        dim_k,
        dim_ff=256,
        dim_d=None,
        dropout_ff=0.15,
        nhead=4,
        depth=1,
        dropout=0.1,
        activation="gelu",
        checkpoint_level="C0",
        parameter_sharing="layerwise",
        k_reduce_by_layer=0,
        full_attention=False,
        include_ff=True,
        w_o_intermediate_dim=None,
        decoder_mode=False,
        causal=False,
        method="learnable",
        ff_intermediate=None,
    ):
        super().__init__()
        assert activation in ["gelu", "swish", "relu"](
            "Only gelu and relu activations supported for now"
        )
        assert (
            checkpoint_level == "C0"
            or checkpoint_level == "C1"
            or checkpoint_level == "C2"
        ), "Checkpoint level has to be either C0, C1, or C2."
        assert (
            parameter_sharing == "none"
            or parameter_sharing == "headwise"
            or parameter_sharing == "kv"
            or parameter_sharing == "layerwise"
        ), (
            "The `parameter_sharing` flag has to be either 'none', 'headwise', 'kv', or 'layerwise'."
        )
        assert channels % nhead == 0 if dim_d is None else True, (
            "If `dim_d` is not set to a custom value, `channels` must be divisible by `nhead`!"
        )
        assert not (ff_intermediate and parameter_sharing == "layerwise"), (
            "Parameter sharing must not be layerwise if ff_intermediate is enabled!"
        )
        assert not (ff_intermediate and decoder_mode), (
            "Raising the dimension in the middle cannot be done in the decoder!"
        )

        layers = torch.nn.ModuleList()
        self.decoder_mode = decoder_mode
        self.input_size = input_size
        self.channels = channels
        self.checkpoint_level = checkpoint_level
        self.depth = depth
        self.nhead = nhead

        head_dim = channels // nhead if dim_d is None else dim_d

        E_proj = get_EF(input_size, dim_k, method, head_dim)
        causal_mask = (
            gen_causal_mask(input_size, dim_k, full_attention)
            if causal
            else None
        )
        # If we want causal but only with the encoder
        causal_enc = (
            gen_causal_mask(input_size, dim_k, full_attention)
            if (causal and not decoder_mode)
            else None
        )

        get_attn = lambda attn_channels, curr_dim_k: MHAttention(
            input_size,
            head_dim,
            attn_channels,
            curr_dim_k,
            nhead,
            dropout,
            checkpoint_level,
            parameter_sharing,
            E_proj,
            E_proj,
            full_attention,
            causal_enc,
            w_o_intermediate_dim,
            decoder_mode=False,
            method=method,
        )
        get_attn_context = lambda attn_channels, curr_dim_k: MHAttention(
            input_size,
            head_dim,
            attn_channels,
            curr_dim_k,
            nhead,
            dropout,
            checkpoint_level,
            parameter_sharing,
            E_proj,
            E_proj,
            full_attention,
            causal_mask,
            w_o_intermediate_dim,
            decoder_mode=True,
            method=method,
        )
        get_ff = lambda input_channels, output_channels: FeedForward(
            input_channels, output_channels, dim_ff, dropout_ff, activation
        )

        for index in range(depth):
            input_channels = (
                ff_intermediate
                if (index != 0 and ff_intermediate is not None)
                and not decoder_mode
                else channels
            )
            output_channels = (
                ff_intermediate
                if (index != depth - 1 and ff_intermediate is not None)
                and not decoder_mode
                else channels
            )
            # TODO: Change the input and output channels here
            attn_layer = get_attn(
                input_channels, max(1, dim_k - index * k_reduce_by_layer)
            )
            ff_layer = get_ff(input_channels, output_channels)

            attn_layer, ff_layer = map(
                lambda res_ch_in, res_ch_out, fn: Residual(
                    fn, res_ch_in, res_ch_out
                ),
                (input_channels, input_channels),
                (input_channels, output_channels),
                (attn_layer, ff_layer),
            )

            if include_ff:
                layers.extend([attn_layer, ff_layer])
            else:
                layers.extend([attn_layer])

            if not self.decoder_mode:
                continue

            attn_context = get_attn_context(
                channels, max(1, dim_k - index * k_reduce_by_layer)
            )
            ff_context = get_ff(channels, channels)

            if not self.decoder_mode:
                continue

            attn_context = get_attn_context(
                channels, max(1, dim_k - index * k_reduce_by_layer)
            )
            ff_context = get_ff(channels, channels)

            attn_context, ff_context = map(
                lambda fn: Residual(fn, channels, channels),
                (attn_context, ff_context),
            )

            if include_ff:
                layers.extend([attn_context, ff_context])
            else:
                layers.extend([attn_context])

        self.seq = layers

    def forward(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len, channels)
        """
        bt, n, c = tensor.shape
        assert n == self.input_size, (
            "This tensor is of the wrong size. Dimension 1 has to match the `input_size` flag"
        )
        assert c == self.channels, (
            "This tensor is of the wrong size. Dimension 2 has to match the `channels` flag"
        )
        assert self.checkpoint_level == "C0" if kwargs else True, (
            "Cannot run checkpointing when using kwargs. Please set the checkpoint level to `C0`"
        )
        assert "embeddings" not in kwargs or self.decoder_mode, (
            "If decoding, needs to be initialized with `decoder_mode=True`"
        )

        for layer in self.seq:
            if self.checkpoint_level != "C0":
                tensor = checkpoint(layer, tensor)
            else:
                tensor = layer(tensor, **kwargs)
        return tensor
