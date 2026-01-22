from typing import Optional

from torch import nn

from ..modeling_outputs import Transformer2DModelOutput
from .transformer_2d import Transformer2DModel

class DualTransformer2DModel(nn.Module):


    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
    ):
        super().__init__()
        self.transformers = nn.ModuleList(
            [
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=in_channels,
                    num_layers=num_layers,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    sample_size=sample_size,
                    num_vector_embeds=num_vector_embeds,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                )
                for _ in range(2)
            ]
        )

        # Variables that can be set by a pipeline:

        # The ratio of transformer1 to transformer2's output states to be combined during inference
        self.mix_ratio = 0.5

        # The shape of `encoder_hidden_states` is expected to be
        # `(batch_size, condition_lengths[0]+condition_lengths[1], num_features)`
        self.condition_lengths = [77, 257]

        # Which transformer to use to encode which condition.
        # E.g. `(1, 0)` means that we'll use `tran...
        self.transformer_index_for_condition = [1, 0]

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        return_dict: bool = True,
    ):
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.
