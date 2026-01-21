# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队
#
# Transformer模块 - 基于注意力的扩散模型

"""
Transformer 模块
包含基于Transformer架构的扩散模型

主要模型：
- Transformer2DModel: 通用2D Transformer
- DiTTransformer2DModel: Diffusion Transformer (DiT)
- PriorTransformer: 先验Transformer

DiT (Diffusion Transformer) 是一种将Transformer架构
应用于扩散模型的方法，在图像生成方面展现出强大性能。
"""

from ...utils import is_torch_available

if is_torch_available():
    from .transformer_2d import Transformer2DModel
    from .dit_transformer_2d import DiTTransformer2DModel
    from .prior_transformer import PriorTransformer
