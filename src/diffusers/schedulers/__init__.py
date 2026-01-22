# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队
#
# 根据Apache许可证2.0版（"许可证"）授权

"""
Diffusers 调度器模块 - 精简版
包含扩散模型的噪声调度算法

主要调度器：
- DDPMScheduler: DDPM调度器（原始的去噪扩散概率模型）
- DDIMScheduler: DDIM调度器（确定性采样，可加速推理）
- ScoreSdeVeScheduler: Score SDE (VE) 调度器（基于分数的SDE方法）
- EulerDiscreteScheduler: Euler离散调度器（常用的ODE求解器）
- EulerAncestralDiscreteScheduler: Euler祖先采样调度器
- PNDMScheduler: PNDM调度器（伪数值方法）
- LMSDiscreteScheduler: LMS离散调度器（线性多步方法）

调度器负责：
1. 定义前向扩散过程（加噪）
2. 定义反向去噪过程
3. 控制采样步数和噪声水平
"""

from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_scipy_available,
    is_torch_available,
)

# 虚拟模块
_dummy_modules = {}
# 导入结构
_import_structure = {}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_pt_objects

    _dummy_modules.update(get_objects_from_module(dummy_pt_objects))
else:
    # 核心调度器
    _import_structure["scheduling_ddim"] = ["DDIMScheduler"]
    _import_structure["scheduling_ddpm"] = ["DDPMScheduler"]
    _import_structure["scheduling_euler_ancestral_discrete"] = ["EulerAncestralDiscreteScheduler"]
    _import_structure["scheduling_euler_discrete"] = ["EulerDiscreteScheduler"]
    _import_structure["scheduling_pndm"] = ["PNDMScheduler"]
    _import_structure["scheduling_sde_ve"] = ["ScoreSdeVeScheduler"]
    _import_structure["scheduling_utils"] = ["KarrasDiffusionSchedulers", "SchedulerMixin"]

try:
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_torch_and_scipy_objects

    _dummy_modules.update(get_objects_from_module(dummy_torch_and_scipy_objects))
else:
    _import_structure["scheduling_lms_discrete"] = ["LMSDiscreteScheduler"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_pt_objects import *
    else:
        # 核心调度器
        from .scheduling_ddim import DDIMScheduler
        from .scheduling_ddpm import DDPMScheduler
        from .scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
        from .scheduling_euler_discrete import EulerDiscreteScheduler
        from .scheduling_pndm import PNDMScheduler
        from .scheduling_sde_ve import ScoreSdeVeScheduler
        from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

    try:
        if not (is_torch_available() and is_scipy_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_torch_and_scipy_objects import *
    else:
        from .scheduling_lms_discrete import LMSDiscreteScheduler

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    for name, value in _dummy_modules.items():
        setattr(sys.modules[__name__], name, value)
