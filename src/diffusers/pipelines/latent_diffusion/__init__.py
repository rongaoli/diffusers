# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队

"""
Latent Diffusion Models (LDM) 管道
潜在扩散模型 - 在潜在空间进行扩散的高效方法

LDM 的核心创新：
1. 使用VAE将图像编码到低维潜在空间
2. 在潜在空间而非像素空间进行扩散
3. 大幅降低计算成本，同时保持生成质量

这是 Stable Diffusion 的基础架构。

参考论文：
"High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
https://arxiv.org/abs/2112.10752
"""

from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_pt_objects

    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
else:
    _import_structure["pipeline_latent_diffusion_superresolution"] = ["LDMSuperResolutionPipeline"]

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_latent_diffusion"] = ["LDMBertModel", "LDMTextToImagePipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_pt_objects import *
    else:
        from .pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline

    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .pipeline_latent_diffusion import LDMBertModel, LDMTextToImagePipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
