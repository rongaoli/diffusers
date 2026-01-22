# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队

"""
DiT (Diffusion Transformer) 管道
扩散Transformer - 使用Transformer架构的扩散模型

DiT 是将 Vision Transformer 架构应用于扩散模型的方法：
1. 用 Transformer 替代 UNet 作为去噪网络
2. 图像被分割为 patches 并嵌入
3. 通过自注意力机制处理全局关系

DiT 展示了 Transformer 在扩散模型中的强大性能，
为后续的大规模图像/视频生成模型奠定了基础。

参考论文：
"Scalable Diffusion Models with Transformers" (Peebles & Xie, 2022)
https://arxiv.org/abs/2212.09748
"""

from typing import TYPE_CHECKING

from ...utils import DIFFUSERS_SLOW_IMPORT, _LazyModule


_import_structure = {"pipeline_dit": ["DiTPipeline"]}

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_dit import DiTPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
