# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队
#
# 根据Apache许可证2.0版（"许可证"）授权

"""
Diffusers 管道模块 - 精简版
包含扩散模型的推理管道

主要组件：
- DDPMPipeline: 去噪扩散概率模型管道（最基础的扩散模型）
- DDIMPipeline: 去噪扩散隐式模型管道（加速采样）
- DiTPipeline: Diffusion Transformer管道
- LDMPipeline: 潜在扩散模型管道
- StableDiffusionPipeline: Stable Diffusion文本到图像管道
"""

from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)

# 虚拟对象（用于处理缺失依赖）
_dummy_objects = {}
# 导入结构
_import_structure = {
    "latent_diffusion": [],
    "stable_diffusion": [],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_pt_objects

    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
else:
    # 基础管道
    _import_structure["ddim"] = ["DDIMPipeline"]
    _import_structure["ddpm"] = ["DDPMPipeline"]
    _import_structure["dit"] = ["DiTPipeline"]
    _import_structure["latent_diffusion"].extend(["LDMSuperResolutionPipeline"])
    _import_structure["pipeline_utils"] = [
        "DiffusionPipeline",
        "ImagePipelineOutput",
    ]

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_torch_and_transformers_objects

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # Latent Diffusion（需要transformers）
    _import_structure["latent_diffusion"].extend(["LDMTextToImagePipeline"])

    # Stable Diffusion 系列
    _import_structure["stable_diffusion"].extend([
        "StableDiffusionPipeline",
        "StableDiffusionImg2ImgPipeline",
        "StableDiffusionInpaintPipeline",
    ])


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_pt_objects import *
    else:
        # 基础扩散管道
        from .ddim import DDIMPipeline
        from .ddpm import DDPMPipeline
        from .dit import DiTPipeline
        from .latent_diffusion import LDMSuperResolutionPipeline
        from .pipeline_utils import (
            DiffusionPipeline,
            ImagePipelineOutput,
        )

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_torch_and_transformers_objects import *
    else:
        # Latent Diffusion
        from .latent_diffusion import LDMTextToImagePipeline

        # Stable Diffusion
        from .stable_diffusion import (
            StableDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
        )

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
