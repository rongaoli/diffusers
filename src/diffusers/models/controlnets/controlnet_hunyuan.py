from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging
from ..attention_processor import AttentionProcessor
from ..embeddings import (
    HunyuanCombinedTimestepTextSizeStyleEmbedding,
    PatchEmbed,
    PixArtAlphaTextProjection,
)
from ..modeling_utils import ModelMixin
from ..transformers.hunyuan_transformer_2d import HunyuanDiTBlock
from .controlnet import Tuple, zero_module

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class HunyuanControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]

class HunyuanDiT2DControlNetModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        conditioning_channels: int = 3,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "gelu-approximate",
        sample_size=32,
        hidden_size=1152,
        transformer_num_layers: int = 40,
        mlp_ratio: float = 4.0,
        cross_attention_dim: int = 1024,
        cross_attention_dim_t5: int = 2048,
        pooled_projection_dim: int = 1024,
        text_len: int = 77,
        text_len_t5: int = 256,
        use_style_cond_and_image_meta_size: bool = True,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim

        self.text_embedder = PixArtAlphaTextProjection(
            in_features=cross_attention_dim_t5,
            hidden_size=cross_attention_dim_t5 * 4,
            out_features=cross_attention_dim,
            act_fn="silu_fp32",
        )

        self.text_embedding_padding = nn.Parameter(
            torch.randn(text_len + text_len_t5, cross_attention_dim, dtype=torch.float32)
        )

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            patch_size=patch_size,
            pos_embed_type=None,
        )

        self.time_extra_emb = HunyuanCombinedTimestepTextSizeStyleEmbedding(
            hidden_size,
            pooled_projection_dim=pooled_projection_dim,
            seq_len=text_len_t5,
            cross_attention_dim=cross_attention_dim_t5,
            use_style_cond_and_image_meta_size=use_style_cond_and_image_meta_size,
        )

        # controlnet_blocks
        self.controlnet_blocks = nn.ModuleList([])

        # HunyuanDiT Blocks
        self.blocks = nn.ModuleList(
            [
                HunyuanDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    activation_fn=activation_fn,
                    ff_inner_dim=int(self.inner_dim * mlp_ratio),
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=True,  # See https://huggingface.co/papers/2302.05442 for details.
                    skip=False,  # always False as it is the first half of the model
                )
                for layer in range(transformer_num_layers // 2 - 1)
            ]
        )
        self.input_block = zero_module(nn.Linear(hidden_size, hidden_size))
        for _ in range(len(self.blocks)):
            controlnet_block = nn.Linear(hidden_size, hidden_size)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
