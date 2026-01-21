# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队
#
# 根据Apache许可证2.0版（"许可证"）授权；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 按"原样"提供，不附带任何明示或暗示的保证或条件。
# 请参阅许可证以了解特定语言的权限和限制。

"""
Diffusers 模型模块 - 精简版
包含扩散模型的核心神经网络架构

主要组件：
- UNet: 2D条件/非条件UNet模型
- AutoencoderKL: 变分自编码器（VAE）
- Transformer: 2D Transformer和DiT模型
- Attention: 注意力机制
"""

from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    _LazyModule,
    is_torch_available,
)

# 导入结构定义
_import_structure = {}

if is_torch_available():
    # 自编码器模块
    _import_structure["autoencoders.autoencoder_kl"] = ["AutoencoderKL"]
    _import_structure["autoencoders.vae"] = ["DiagonalGaussianDistribution", "Encoder", "Decoder"]

    # 嵌入层
    _import_structure["embeddings"] = ["ImageProjection", "TimestepEmbedding", "Timesteps"]

    # 模型基类和工具
    _import_structure["modeling_utils"] = ["ModelMixin"]
    _import_structure["modeling_outputs"] = ["AutoencoderKLOutput", "Transformer2DModelOutput"]

    # Transformer模块
    _import_structure["transformers.transformer_2d"] = ["Transformer2DModel"]
    _import_structure["transformers.dit_transformer_2d"] = ["DiTTransformer2DModel"]
    _import_structure["transformers.prior_transformer"] = ["PriorTransformer"]

    # UNet模块
    _import_structure["unets.unet_2d"] = ["UNet2DModel"]
    _import_structure["unets.unet_2d_condition"] = ["UNet2DConditionModel"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        # 自编码器
        from .autoencoders import (
            AutoencoderKL,
        )
        from .autoencoders.vae import DiagonalGaussianDistribution, Encoder, Decoder

        # 嵌入层
        from .embeddings import ImageProjection, TimestepEmbedding, Timesteps

        # 模型基类
        from .modeling_utils import ModelMixin
        from .modeling_outputs import AutoencoderKLOutput, Transformer2DModelOutput

        # Transformer
        from .transformers import (
            Transformer2DModel,
            DiTTransformer2DModel,
            PriorTransformer,
        )

        # UNet
        from .unets import (
            UNet2DModel,
            UNet2DConditionModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
