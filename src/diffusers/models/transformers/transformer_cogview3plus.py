from typing import Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import AttentionMixin, FeedForward
from ..attention_processor import Attention, CogVideoXAttnProcessor2_0
from ..embeddings import CogView3CombinedTimestepSizeEmbeddings, CogView3PlusPatchEmbed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, CogView3PlusAdaLayerNormZeroTextImage

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class CogView3PlusTransformerBlock(nn.Module):


    def __init__(
        self,
        dim: int = 2560,
        num_attention_heads: int = 64,
        attention_head_dim: int = 40,
        time_embed_dim: int = 512,
    ):
        super().__init__()

        self.norm1 = CogView3PlusAdaLayerNormZeroTextImage(embedding_dim=time_embed_dim, dim=dim)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=dim,
            bias=True,
            qk_norm="layer_norm",
            elementwise_affine=False,
            eps=1e-6,
            processor=CogVideoXAttnProcessor2_0(),
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)

        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1(hidden_states, encoder_hidden_states, emb)

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
        )

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * ff_output[:, :text_seq_length]

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return hidden_states, encoder_hidden_states

class CogView3PlusTransformer2DModel(ModelMixin, AttentionMixin, ConfigMixin):


    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _no_split_modules = ["CogView3PlusTransformerBlock", "CogView3PlusPatchEmbed"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 30,
        attention_head_dim: int = 40,
        num_attention_heads: int = 64,
        out_channels: int = 16,
        text_embed_dim: int = 4096,
        time_embed_dim: int = 512,
        condition_dim: int = 256,
        pos_embed_max_size: int = 128,
        sample_size: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # CogView3 uses 3 additional SDXL-like conditions - original_size, target_size, crop_coords
        # Each of these are sincos embeddings of shape 2 * condition_dim
        self.pooled_projection_dim = 3 * 2 * condition_dim

        self.patch_embed = CogView3PlusPatchEmbed(
            in_channels=in_channels,
            hidden_size=self.inner_dim,
            patch_size=patch_size,
            text_hidden_size=text_embed_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.time_condition_embed = CogView3CombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=self.pooled_projection_dim,
            timesteps_dim=self.inner_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                CogView3PlusTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=time_embed_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        original_size: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        The [`CogView3PlusTransformer2DModel`] forward method.
                `(batch_size, sequence_len, text_embed_dim)`
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            original_size (`torch.Tensor`):
                CogView3 uses SDXL-like micro-conditioning for original image size as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`torch.Tensor`):
                CogView3 uses SDXL-like micro-conditioning for target image size as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crop_coords (`torch.Tensor`):
                CogView3 uses SDXL-like micro-conditioning for crop coordinates as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
