import torch
import torch.nn as nn

"""
下采样层模块

用于在扩散模型（特别是 UNet）中减少特征图的空间分辨率
"""

class Downsample1D(nn.Module):
    """
    1D 下采样层

    参数:
        channels: 输入/输出通道数
        use_conv: True 则使用卷积，False 则使用平均池化
    """

    def __init__(self, channels: int, use_conv: bool = False, out_channels: int = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        if use_conv:
            # 步长为2的卷积实现下采样
            self.conv = nn.Conv1d(channels, self.out_channels, 3, stride=2, padding=1)
        else:
            # 平均池化下采样
            self.conv = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


class Downsample2D(nn.Module):
    """
    2D 下采样层

    常用于 UNet 的编码器部分
    两种方式：
    1. 步长卷积（更常用）
    2. 平均池化
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int = None,
        padding: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        if use_conv:
            # 步长为2的卷积
            self.conv = nn.Conv2d(channels, self.out_channels, 3, stride=2, padding=padding)
        else:
            # 平均池化
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# 概念说明
# ============================================================================
"""
下采样 (Downsampling) 的两种主要方法：

1. 步长卷积 (Strided Convolution) - 更常用
   - 使用 stride=2 的卷积直接减少空间分辨率
   - 同时可以改变通道数
   - 可学习的下采样方式

2. 池化 (Pooling)
   - MaxPool: 取窗口内最大值
   - AvgPool: 取窗口内平均值
   - 非参数化，不需要学习

在 UNet 编码器中的作用：
- 逐步减少空间分辨率
- 增加感受野，捕获更大范围的上下文信息
- 通常伴随通道数的增加（如：64→128→256→512）

典型的 UNet 编码器结构：
输入 (H×W×C)
  ↓ ResBlock + Downsample
中间 (H/2×W/2×2C)
  ↓ ResBlock + Downsample
中间 (H/4×W/4×4C)
  ↓ ResBlock + Downsample
瓶颈 (H/8×W/8×8C)
"""
