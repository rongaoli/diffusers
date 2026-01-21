# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队
#
# 根据Apache许可证2.0版（"许可证"）授权

"""
Diffusers 库 - 精简学习版
=====================================

这是 HuggingFace Diffusers 库的精简版本，
专门为学习和理解扩散模型（Diffusion Models）而设计。

核心概念：
---------
扩散模型是一类生成模型，通过以下两个过程工作：
1. 前向扩散过程：逐步向数据添加噪声
2. 反向去噪过程：学习从噪声恢复原始数据

主要组件：
---------
1. Models（模型）：神经网络架构
   - UNet2DModel: 基础无条件UNet
   - UNet2DConditionModel: 条件UNet（用于文本到图像）
   - AutoencoderKL: 变分自编码器（VAE）
   - DiTTransformer2DModel: Diffusion Transformer

2. Schedulers（调度器）：控制噪声添加和去除
   - DDPMScheduler: 原始DDPM调度器
   - DDIMScheduler: 加速采样的DDIM调度器
   - EulerDiscreteScheduler: Euler离散调度器
   - ScoreSdeVeScheduler: Score SDE调度器

3. Pipelines（管道）：端到端的推理流程
   - DDPMPipeline: 基础DDPM管道
   - DDIMPipeline: DDIM管道
   - DiTPipeline: DiT管道
   - LDMTextToImagePipeline: 潜在扩散文本到图像
   - StableDiffusionPipeline: Stable Diffusion管道

快速开始：
--------
```python
from diffusers import DDPMPipeline

# 加载预训练模型
pipeline = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

# 生成图像
image = pipeline().images[0]
image.save("generated_cat.png")
```

学习资源：
--------
- DDPM论文: https://arxiv.org/abs/2006.11239
- DDIM论文: https://arxiv.org/abs/2010.02502
- LDM论文: https://arxiv.org/abs/2112.10752
- DiT论文: https://arxiv.org/abs/2212.09748
"""

__version__ = "0.37.0.dev0-simplified"

from typing import TYPE_CHECKING

from .utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_scipy_available,
    is_torch_available,
    is_transformers_available,
    logging,
)


# 延迟导入结构
# 基于 https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py
_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "loaders": [],
    "models": [],
    "pipelines": [],
    "schedulers": [],
    "utils": [
        "OptionalDependencyNotAvailable",
        "is_scipy_available",
        "is_torch_available",
        "is_transformers_available",
        "logging",
    ],
}

# ==================== PyTorch 依赖 ====================
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [
        name for name in dir(dummy_pt_objects) if not name.startswith("_")
    ]
else:
    # 模型
    _import_structure["models"].extend([
        "AutoencoderKL",
        "DiTTransformer2DModel",
        "ModelMixin",
        "PriorTransformer",
        "Transformer2DModel",
        "UNet2DConditionModel",
        "UNet2DModel",
    ])

    # 调度器
    _import_structure["schedulers"].extend([
        "DDIMScheduler",
        "DDPMScheduler",
        "EulerAncestralDiscreteScheduler",
        "EulerDiscreteScheduler",
        "KarrasDiffusionSchedulers",
        "PNDMScheduler",
        "SchedulerMixin",
        "ScoreSdeVeScheduler",
    ])

    # 基础管道
    _import_structure["pipelines"].extend([
        "DDIMPipeline",
        "DDPMPipeline",
        "DiffusionPipeline",
        "DiTPipeline",
        "ImagePipelineOutput",
        "LDMSuperResolutionPipeline",
    ])

    # 加载器
    _import_structure["loaders"].extend([
        "UNet2DConditionLoadersMixin",
    ])

# ==================== PyTorch + SciPy 依赖 ====================
try:
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_scipy_objects

    _import_structure["utils.dummy_torch_and_scipy_objects"] = [
        name for name in dir(dummy_torch_and_scipy_objects) if not name.startswith("_")
    ]
else:
    _import_structure["schedulers"].extend([
        "LMSDiscreteScheduler",
    ])

# ==================== PyTorch + Transformers 依赖 ====================
try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_transformers_objects

    _import_structure["utils.dummy_torch_and_transformers_objects"] = [
        name for name in dir(dummy_torch_and_transformers_objects) if not name.startswith("_")
    ]
else:
    # Latent Diffusion 和 Stable Diffusion 管道
    _import_structure["pipelines"].extend([
        "LDMTextToImagePipeline",
        "StableDiffusionImg2ImgPipeline",
        "StableDiffusionInpaintPipeline",
        "StableDiffusionPipeline",
    ])


# ==================== 类型检查导入 ====================
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .configuration_utils import ConfigMixin

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_pt_objects import *
    else:
        # 模型
        from .models import (
            AutoencoderKL,
            DiTTransformer2DModel,
            ModelMixin,
            PriorTransformer,
            Transformer2DModel,
            UNet2DConditionModel,
            UNet2DModel,
        )

        # 调度器
        from .schedulers import (
            DDIMScheduler,
            DDPMScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            KarrasDiffusionSchedulers,
            PNDMScheduler,
            SchedulerMixin,
            ScoreSdeVeScheduler,
        )

        # 管道
        from .pipelines import (
            DDIMPipeline,
            DDPMPipeline,
            DiffusionPipeline,
            DiTPipeline,
            ImagePipelineOutput,
            LDMSuperResolutionPipeline,
        )

        # 加载器
        from .loaders import UNet2DConditionLoadersMixin

    try:
        if not (is_torch_available() and is_scipy_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_scipy_objects import *
    else:
        from .schedulers import LMSDiscreteScheduler

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_transformers_objects import *
    else:
        from .pipelines import (
            LDMTextToImagePipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionPipeline,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
