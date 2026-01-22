import torch
import torch.nn.functional as F
from torch import nn

"""
激活函数模块

扩散模型中常用的各种激活函数实现
"""

# 激活函数映射表
ACT2CLS = {
    "swish": nn.SiLU,  # Swish 等同于 SiLU
    "silu": nn.SiLU,   # Sigmoid Linear Unit
    "mish": nn.Mish,
    "gelu": nn.GELU,   # Gaussian Error Linear Unit
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    """根据名称获取激活函数"""
    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"未知激活函数: {act_fn}")


class GELU(nn.Module):
    """
    GELU 激活函数（带线性投影）

    GELU(x) = x * Φ(x)，其中 Φ 是标准正态分布的累积分布函数
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate  # "none" 或 "tanh" (更快但近似)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate=self.approximate)


class GEGLU(nn.Module):
    """
    门控 GELU (Gated GELU)

    论文: https://arxiv.org/abs/2002.05202
    输出: hidden_states ⊙ GELU(gate)
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        # 输出维度 x2，因为需要拆分为 hidden_states 和 gate
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        # 拆分为两部分
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        # 门控机制：用 GELU(gate) 调制 hidden_states
        return hidden_states * F.gelu(gate)


class SwiGLU(nn.Module):
    """
    门控 SiLU/Swish (Gated SiLU)

    类似 GEGLU，但使用 SiLU 代替 GELU
    输出: hidden_states ⊙ SiLU(gate)
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class ApproximateGELU(nn.Module):
    """
    GELU 的近似形式

    GELU(x) ≈ x * sigmoid(1.702 * x)
    比标准 GELU 计算更快
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


# 说明：
# 门控激活函数 (GEGLU, SwiGLU) 的优势：
# - 提供更强的表达能力
# - 允许网络选择性地传递信息
# - 在 Transformer 架构中特别有效
