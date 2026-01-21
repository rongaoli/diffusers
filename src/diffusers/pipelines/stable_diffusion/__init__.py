# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队

"""
Stable Diffusion 管道
稳定扩散模型 - 目前最流行的文本到图像生成模型

Stable Diffusion 基于潜在扩散模型（LDM），主要特点：
1. 使用 CLIP 文本编码器理解文本提示
2. 使用 VAE 在潜在空间进行扩散
3. 使用 UNet 预测噪声
4. 支持多种任务：文本到图像、图像到图像、图像修复

主要管道：
- StableDiffusionPipeline: 文本到图像生成
- StableDiffusionImg2ImgPipeline: 图像到图像转换
- StableDiffusionInpaintPipeline: 图像修复/补全

参考：
https://stability.ai/stable-diffusion
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
_import_structure = {"pipeline_output": ["StableDiffusionPipelineOutput"]}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_stable_diffusion"] = ["StableDiffusionPipeline"]
    _import_structure["pipeline_stable_diffusion_img2img"] = ["StableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint"] = ["StableDiffusionInpaintPipeline"]
    _import_structure["safety_checker"] = ["StableDiffusionSafetyChecker"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *

    else:
        from .pipeline_output import StableDiffusionPipelineOutput
        from .pipeline_stable_diffusion import StableDiffusionPipeline
        from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
        from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
        from .safety_checker import StableDiffusionSafetyChecker

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
