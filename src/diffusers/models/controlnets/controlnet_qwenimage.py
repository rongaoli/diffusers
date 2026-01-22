from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from ..attention import AttentionMixin
from ..cache_utils import CacheMixin
from ..controlnets.controlnet import zero_module
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..transformers.transformer_qwenimage import (
    QwenEmbedRope,
    QwenImageTransformerBlock,
    QwenTimestepProjEmbeddings,
    RMSNorm,
    compute_text_seq_len_from_mask,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class QwenImageControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]

class QwenImageControlNetModel(
    ModelMixin, AttentionMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        extra_condition_channels: int = 0,  # for controlnet-inpainting
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
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
            self.controlnet_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))
        self.controlnet_x_embedder = zero_module(
            torch.nn.Linear(in_channels + extra_condition_channels, self.inner_dim)
        )

        self.gradient_checkpointing = False

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = 5,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        load_weights_from_transformer=True,
        extra_condition_channels: int = 0,
    ):
        config = dict(transformer.config)
        config["num_layers"] = num_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads
        config["extra_condition_channels"] = extra_condition_channels

        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            controlnet.img_in.load_state_dict(transformer.img_in.state_dict())
            controlnet.txt_in.load_state_dict(transformer.txt_in.state_dict())
            controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)
            controlnet.controlnet_x_embedder = zero_module(controlnet.controlnet_x_embedder)

        return controlnet

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        The [`QwenImageControlNetModel`] forward method.
                (not just contiguous valid tokens followed by padding) since it's applied element-wise in attention.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            img_shapes (`List[Tuple[int, int, int]]`, *optional*):
                Image shapes for RoPE computation.
            txt_seq_lens (`List[int]`, *optional*):
                **Deprecated**. Not needed anymore, we use `encoder_hidden_states` instead to infer text sequence
                length.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.
