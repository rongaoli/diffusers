from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..attention import AttentionMixin, JointTransformerBlock
from ..attention_processor import Attention, FusedJointAttnProcessor2_0
from ..embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..transformers.transformer_sd3 import SD3SingleTransformerBlock
from .controlnet import BaseOutput, zero_module

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class SD3ControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]

class SD3ControlNetModel(ModelMixin, AttentionMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    class SD3ControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]

class SD3ControlNetModel(ModelMixin, AttentionMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):


    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        extra_conditioning_channels: int = 0,
        dual_attention_layers: Tuple[int, ...] = (),
        qk_norm: Optional[str] = None,
        pos_embed_type: Optional[str] = "sincos",
        use_pos_embed: bool = True,
        force_zeros_for_pooled_projection: bool = True,
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        if use_pos_embed:
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=self.inner_dim,
                pos_embed_max_size=pos_embed_max_size,
                pos_embed_type=pos_embed_type,
            )
        else:
            self.pos_embed = None
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        if joint_attention_dim is not None:
            self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)

            # `attention_head_dim` is doubled to account for the mixing.
            # It needs to crafted when we get the actual checkpoints.
            self.transformer_blocks = nn.ModuleList(
                [
                    JointTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        context_pre_only=False,
                        qk_norm=qk_norm,
                        use_dual_attention=True if i in dual_attention_layers else False,
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.context_embedder = None
            self.transformer_blocks = nn.ModuleList(
                [
                    SD3SingleTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                    )
                    for _ in range(num_layers)
                ]
            )

        # controlnet_blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            controlnet_block = nn.Linear(self.inner_dim, self.inner_dim)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)
        pos_embed_input = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels + extra_conditioning_channels,
            embed_dim=self.inner_dim,
            pos_embed_type=None,
        )
        self.pos_embed_input = zero_module(pos_embed_input)

        self.gradient_checkpointing = False

    # Copied from diffusers.models.unets.unet_3d_c...
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:

        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.transformers.tr...
    def fuse_qkv_projections(self):

        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedJointAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_c...
    def unfuse_qkv_projections(self):

        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    # Notes: This is for SD3.5 8b controlnet, which shares the pos_embed with the transformer
    # we should have handled this in conversion script
    def _get_pos_embed_from_transformer(self, transformer):
        pos_embed = PatchEmbed(
            height=transformer.config.sample_size,
            width=transformer.config.sample_size,
            patch_size=transformer.config.patch_size,
            in_channels=transformer.config.in_channels,
            embed_dim=transformer.inner_dim,
            pos_embed_max_size=transformer.config.pos_embed_max_size,
        )
        pos_embed.load_state_dict(transformer.pos_embed.state_dict(), strict=True)
        return pos_embed

    @classmethod
    def from_transformer(
        cls, transformer, num_layers=12, num_extra_conditioning_channels=1, load_weights_from_transformer=True
    ):
        config = transformer.config
        config["num_layers"] = num_layers or config.num_layers
        config["extra_conditioning_channels"] = num_extra_conditioning_channels
        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)

            controlnet.pos_embed_input = zero_module(controlnet.pos_embed_input)

        return controlnet

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        The [`SD3Transformer2DModel`] forward method.
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
