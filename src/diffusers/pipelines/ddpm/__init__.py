# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队

"""
DDPM (Denoising Diffusion Probabilistic Models) 管道
去噪扩散概率模型 - 扩散模型的基础架构

DDPM 是扩散模型的开创性工作，核心思想：
1. 前向过程：逐步向数据添加高斯噪声，直到变成纯噪声
2. 反向过程：训练神经网络学习去噪，从噪声恢复数据

参考论文：
"Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
https://arxiv.org/abs/2006.11239
"""

from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    _LazyModule,
)


_import_structure = {"pipeline_ddpm": ["DDPMPipeline"]}

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_ddpm import DDPMPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
