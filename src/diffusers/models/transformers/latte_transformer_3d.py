from typing import Optional

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ..attention import BasicTransformerBlock
from ..cache_utils import CacheMixin
from ..embeddings import PatchEmbed, PixArtAlphaTextProjection, get_1d_sincos_pos_embed_from_grid
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormSingle

class LatteTransformer3DModel(ModelMixin, ConfigMixin, CacheMixin):
    _supports_gradient_checkpointing = True


    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: int = 64,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        caption_channels: int = None,
        video_length: int = 16,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Define input layers
        self.height = sample_size
        self.width = sample_size

        interpolation_scale = self.config.sample_size // 64
        interpolation_scale = max(interpolation_scale, 1)
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
        )

        # 2. Define spatial transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for d in range(num_layers)
            ]
        )

        # 3. Define temporal transformers blocks
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=None,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        # 5. Latte other blocks.
        self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)
        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        # define temporal positional embedding
        temp_pos_embed = get_1d_sincos_pos_embed_from_grid(
            inner_dim, torch.arange(0, video_length).unsqueeze(1), output_type="pt"
        )  # 1152 hidden size
        self.register_buffer("temp_pos_embed", temp_pos_embed.float().unsqueeze(0), persistent=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):
        The [`LatteTransformer3DModel`] forward method.

                    * Mask `(batcheight, sequence_length)` True = keep, False = discard.
                    * Bias `(batcheight, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            enable_temporal_attentions:
                (`bool`, *optional*, defaults to `True`): Whether to enable temporal attentions.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.
