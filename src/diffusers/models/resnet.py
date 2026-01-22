import torch
import torch.nn as nn
from typing import Optional

from .activations import get_activation
from .normalization import AdaGroupNorm

"""
ResNet 块模块

ResNet 块是 UNet 的基础构建单元
核心思想：残差连接（Skip Connection）
"""

class ResnetBlock2D(nn.Module):
    """
    标准 ResNet 块（用于扩散模型 UNet）

    结构：
    x -> [GroupNorm -> Activation -> Conv] -> [GroupNorm -> Activation -> Conv] -> + x
         |                                                                          |
         +--------------------------------------------------------------------------+
                                    残差连接（shortcut）

    关键特性：
    - 时间嵌入注入（通过 AdaGroupNorm）
    - 残差连接帮助梯度流动
    - GroupNorm 稳定训练
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,        # 时间嵌入维度
        groups: int = 32,                 # GroupNorm 分组数
        eps: float = 1e-6,
        non_linearity: str = "swish",     # 激活函数
        output_scale_factor: float = 1.0,  # 输出缩放（用于某些架构）
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # 第一个归一化层（带时间嵌入）
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 时间嵌入投影（将 temb 投影到 out_channels）
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        # 第二个归一化和卷积
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 激活函数
        self.nonlinearity = get_activation(non_linearity)

        # 如果输入输出通道数不同，需要一个投影层用于残差连接
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

        self.output_scale_factor = output_scale_factor

    def forward(self, input_tensor: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        参数:
            input_tensor: 输入特征 (B, C, H, W)
            temb: 时间嵌入 (B, temb_channels)

        返回:
            输出特征 (B, out_channels, H, W)
        """
        hidden_states = input_tensor

        # 1. 第一组：Norm -> Act -> Conv
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # 2. 注入时间嵌入（如果有）
        if temb is not None and self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]  # (B, C, 1, 1)
            hidden_states = hidden_states + temb

        # 3. 第二组：Norm -> Act -> Conv
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        # 4. 残差连接
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class ResnetBlock2DCondNorm(nn.Module):
    """
    带条件归一化的 ResNet 块

    使用 AdaGroupNorm 而非普通 GroupNorm
    时间嵌入通过调制 norm 的 scale 和 shift 注入
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        # 使用 AdaGroupNorm（自适应组归一化）
        self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        AdaGroupNorm 通过 temb 调制归一化：
        norm_out = scale(temb) * GroupNorm(x) + shift(temb)
        """
        residual = hidden_states

        # 第一组
        hidden_states = self.norm1(hidden_states, temb)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # 第二组
        hidden_states = self.norm2(hidden_states, temb)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        # 残差连接
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return residual + hidden_states


# ============================================================================
# 核心概念说明
# ============================================================================
"""
ResNet 块在扩散模型中的作用：

1. 残差连接（Skip Connection）
   为什么需要？
   - 解决深度网络的梯度消失问题
   - 允许网络学习"增量"而非完整映射
   - 公式：output = F(x) + x

2. 时间嵌入注入
   两种方式：
   a) 直接加法（ResnetBlock2D）:
      hidden = conv(norm(x)) + time_proj(temb)

   b) 调制归一化（ResnetBlock2DCondNorm）:
      hidden = (scale(temb) + 1) * GroupNorm(x) + shift(temb)
      这种方式更强大，允许时间信息调制特征的统计量

3. GroupNorm
   为什么不用 BatchNorm？
   - BatchNorm 在小 batch 时不稳定
   - GroupNorm 独立于 batch size
   - 更适合扩散模型（通常 batch 较小）

4. 在 UNet 中的使用
   UNet 编码器：
   - ResBlock -> Downsample -> ResBlock -> Downsample -> ...

   UNet 瓶颈层：
   - ResBlock -> Attention -> ResBlock

   UNet 解码器：
   - ResBlock -> Upsample -> ResBlock -> Upsample -> ...
   - 每层还有来自编码器的跳跃连接

5. 典型配置
   - 通道数：64, 128, 256, 512 (逐层增加)
   - GroupNorm 分组：32（通常）
   - 激活函数：SiLU/Swish（现代架构）或 ReLU（旧架构）
   - 时间嵌入维度：512 或更大

6. ResNet vs Transformer
   ResNet 优势：
   - 参数效率高
   - 归纳偏置强（适合图像）
   - 计算效率高

   Transformer 优势：
   - 建模长距离依赖
   - 灵活性更强

   实践中：两者结合使用
   - ResNet 处理局部特征
   - Attention/Transformer 处理全局依赖
"""
