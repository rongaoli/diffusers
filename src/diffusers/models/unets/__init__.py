# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队
#
# UNet模块 - 扩散模型的核心骨干网络

"""
UNet 模块
包含扩散模型中使用的U-Net架构

主要模型：
- UNet2DModel: 基础2D UNet，用于无条件图像生成
- UNet2DConditionModel: 条件2D UNet，用于条件图像生成（如文本到图像）

U-Net架构特点：
1. 编码器-解码器结构
2. 跳跃连接（skip connections）
3. 时间步嵌入（用于去噪过程）
4. 可选的交叉注意力（用于条件生成）
"""

from ...utils import is_torch_available

if is_torch_available():
    from .unet_2d import UNet2DModel
    from .unet_2d_condition import UNet2DConditionModel
