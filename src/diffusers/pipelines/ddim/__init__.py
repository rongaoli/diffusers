# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队

"""
DDIM (Denoising Diffusion Implicit Models) 管道
去噪扩散隐式模型 - 加速采样的扩散模型

DDIM 是 DDPM 的改进版本，主要优势：
1. 确定性采样：给定相同的初始噪声，总是生成相同的结果
2. 加速推理：可以跳过步骤，用更少的步数完成采样
3. 语义插值：可以在潜在空间进行有意义的插值

参考论文：
"Denoising Diffusion Implicit Models" (Song et al., 2020)
https://arxiv.org/abs/2010.02502
"""

from typing import TYPE_CHECKING

from ...utils import DIFFUSERS_SLOW_IMPORT, _LazyModule


_import_structure = {"pipeline_ddim": ["DDIMPipeline"]}

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_ddim import DDIMPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
