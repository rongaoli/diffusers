# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队

"""
Diffusers 加载器模块 - 精简版
包含模型加载的基础工具

主要功能：
- UNet2DConditionLoadersMixin: UNet加载器混入类
- AttnProcsLayers: 注意力处理层

注意：此精简版移除了LoRA、PEFT、IP-Adapter等高级功能。
"""

from typing import TYPE_CHECKING

from ..utils import DIFFUSERS_SLOW_IMPORT, _LazyModule
from ..utils.import_utils import is_torch_available


_import_structure = {}

if is_torch_available():
    _import_structure["unet"] = ["UNet2DConditionLoadersMixin"]
    _import_structure["utils"] = ["AttnProcsLayers"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from .unet import UNet2DConditionLoadersMixin
        from .utils import AttnProcsLayers

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
