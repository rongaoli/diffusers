import torch
import torch.nn as nn
import torch.nn.functional as F

"""
上采样层模块

用于在扩散模型（特别是 UNet）中增加特征图的空间分辨率
"""

class Upsample1D(nn.Module):
    """
    1D 上采样层

    参数:
        channels: 输入/输出通道数
        use_conv: 是否使用卷积（而非简单插值）
        use_conv_transpose: 是否使用转置卷积
    """

    def __init__(self, channels: int, use_conv: bool = False, use_conv_transpose: bool = False):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        self.conv = None
        if use_conv_transpose:
            # 转置卷积：kernel=4, stride=2, padding=1
            self.conv = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        elif use_conv:
            # 普通卷积（配合插值使用）
            self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        if self.use_conv_transpose:
            return self.conv(x)

        # 最近邻插值 2倍上采样
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x


class Upsample2D(nn.Module):
    """
    2D 上采样层

    常用于 UNet 的解码器部分
    两种方式：
    1. 插值 + 卷积（更常用）
    2. 转置卷积
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: int = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        if use_conv_transpose:
            self.conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv2d(channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        if self.use_conv_transpose:
            return self.conv(x)

        # 最近邻插值 2倍上采样
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x


# ============================================================================
# 概念说明
# ============================================================================
"""
上采样 (Upsampling) 的两种主要方法：

1. 插值 + 卷积 (更常用)
   - 先用插值（nearest/bilinear）增加空间分辨率
   - 再用卷积细化特征
   - 优点：参数少，不容易产生棋盘伪影

2. 转置卷积 (Transposed Convolution)
   - 也称为反卷积 (Deconvolution)
   - 直接学习上采样过程
   - 缺点：可能产生棋盘伪影 (checkerboard artifacts)

在 UNet 解码器中的作用：
- 将低分辨率特征图逐步恢复到原始分辨率
- 通常与编码器的跳跃连接 (skip connection) 结合使用
"""
