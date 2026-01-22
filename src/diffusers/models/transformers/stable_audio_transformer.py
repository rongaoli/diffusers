from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import AttentionMixin, FeedForward
from ..attention_processor import Attention, StableAudioAttnProcessor2_0
from ..modeling_utils import ModelMixin
from ..transformers.transformer_2d import Transformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class StableAudioGaussianFourierProjection(nn.Module):


    # Copied from diffusers.models.embeddings.GaussianFourierProjection.__init__
    def __init__(
        self, embedding_size: int = 256, scale: float = 1.0, set_W_to_weight=True, log=True, flip_sin_to_cos=False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

        if set_W_to_weight:
            # to delete later
            del self.weight
            self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
            self.weight = self.W
            del self.W

    def forward(self, x):
        if self.log:
            x = torch.log(x)

        x_proj = 2 * np.pi * x[:, None] @ self.weight[None, :]

        if self.flip_sin_to_cos:
            out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        else:
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out

@maybe_allow_in_graph
class StableAudioDiTBlock(nn.Module):


    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_key_value_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        upcast_attention: bool = False,
        norm_eps: float = 1e-5,
        ff_inner_dim: Optional[int] = None,
    ):
        super().__init__()
        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
            out_bias=False,
            processor=StableAudioAttnProcessor2_0(),
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, norm_eps, True)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_key_value_attention_heads,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
            out_bias=False,
            processor=StableAudioAttnProcessor2_0(),
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, True)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn="swiglu",
            final_dropout=False,
            inner_dim=ff_inner_dim,
            bias=True,
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        rotary_embedding: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # Notice that normalization is always appl...
        # 0. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_embedding,
        )

        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states

        return hidden_states

class StableAudioDiTModel(ModelMixin, AttentionMixin, ConfigMixin):


    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["preprocess_conv", "postprocess_conv", "^proj_in$", "^proj_out$", "norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 1024,
        in_channels: int = 64,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        num_key_value_attention_heads: int = 12,
        out_channels: int = 64,
        cross_attention_dim: int = 768,
        time_proj_dim: int = 256,
        global_states_input_dim: int = 1536,
        cross_attention_input_dim: int = 768,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.time_proj = StableAudioGaussianFourierProjection(
            embedding_size=time_proj_dim // 2,
            flip_sin_to_cos=True,
            log=False,
            set_W_to_weight=False,
        )

        self.timestep_proj = nn.Sequential(
            nn.Linear(time_proj_dim, self.inner_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim, bias=True),
        )

        self.global_proj = nn.Sequential(
            nn.Linear(global_states_input_dim, self.inner_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim, bias=False),
        )

        self.cross_attention_proj = nn.Sequential(
            nn.Linear(cross_attention_input_dim, cross_attention_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cross_attention_dim, cross_attention_dim, bias=False),
        )

        self.preprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.proj_in = nn.Linear(in_channels, self.inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                StableAudioDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_key_value_attention_heads=num_key_value_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(self.out_channels, self.out_channels, 1, bias=False)

        self.gradient_checkpointing = False

    # Copied from diffusers.models.transformers.hu...
    def set_default_attn_processor(self):

        self.set_attn_processor(StableAudioAttnProcessor2_0())

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        global_hidden_states: torch.FloatTensor = None,
        rotary_embedding: torch.FloatTensor = None,
        return_dict: bool = True,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        The [`StableAudioDiTModel`] forward method.

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token cross-attention indices, formed by concatenating
                the attention masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
