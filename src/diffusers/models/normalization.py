import torch
import torch.nn as nn
import torch.nn.functional as F

"""
归一化层模块

扩散模型中使用的各种归一化技术，特别是自适应归一化（Adaptive Normalization）
"""

class AdaLayerNorm(nn.Module):
    """
    自适应 Layer Normalization (AdaLN)

    核心思想：用时间步嵌入 (timestep embedding) 来调制归一化的 scale 和 shift
    公式: AdaLN(x, t) = scale(t) * LayerNorm(x) + shift(t)

    用于：DiT 等基于 Transformer 的扩散模型
    """

    def __init__(self, embedding_dim: int, num_embeddings = None, output_dim = None):
        super().__init__()
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, elementwise_affine=False)

    def forward(self, x, timestep=None, temb=None):
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=0)

        return self.norm(x) * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    """
    AdaLN-Zero: 带零初始化的自适应 LayerNorm

    来自 DiT 论文，输出额外的门控参数用于残差连接
    输出: (norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        # 拆分为 6 个参数：MSA 的 shift/scale/gate + MLP 的 shift/scale/gate
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaGroupNorm(nn.Module):
    """
    自适应 Group Normalization

    类似 AdaLN，但用于卷积层（UNet）
    通过时间嵌入调制 GroupNorm 的 scale 和 shift
    """

    def __init__(self, embedding_dim: int, out_dim: int, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups
        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x, emb):
        emb = self.linear(emb)
        emb = emb[:, :, None, None]  # 扩展为 (B, C, 1, 1)
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups)
        return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    """
    RMS (Root Mean Square) Normalization

    论文: https://arxiv.org/abs/1910.07467
    相比 LayerNorm 更简单高效，不需要减去均值
    公式: RMSNorm(x) = x / RMS(x) * scale
         其中 RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()

        # 计算 RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # 应用可学习的 scale
        hidden_states = hidden_states.to(input_dtype) * self.weight
        return hidden_states


class LayerNorm(nn.LayerNorm):
    """
    标准 Layer Normalization

    这里主要是为了兼容性包装
    """
    pass


class GlobalResponseNorm(nn.Module):
    """
    全局响应归一化 (GRN)

    来自 ConvNeXt-v2: https://arxiv.org/abs/2301.00808
    用于增强卷积网络中的通道间竞争
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # 计算空间范数
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # 归一化
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        # 门控 + 残差
        return self.gamma * (x * nx) + self.beta + x


def get_normalization(norm_type: str, num_features: int, eps: float = 1e-5):
    """根据类型名称获取归一化层"""
    if norm_type == "rms_norm":
        return RMSNorm(num_features, eps=eps)
    elif norm_type == "layer_norm":
        return nn.LayerNorm(num_features, eps=eps)
    elif norm_type == "batch_norm":
        return nn.BatchNorm2d(num_features, eps=eps)
    else:
        raise ValueError(f"未知归一化类型: {norm_type}")


# ============================================================================
# 核心概念说明
# ============================================================================
"""
自适应归一化 (Adaptive Normalization) 的核心思想：

传统归一化：
    output = (x - mean) / std * scale + bias
    其中 scale 和 bias 是可学习参数，对所有样本固定

自适应归一化：
    output = (x - mean) / std * scale(condition) + shift(condition)
    其中 scale 和 shift 由条件信息（如时间步）动态生成

优势：
1. 允许模型根据条件（时间步、类别等）调整特征分布
2. 在扩散模型中特别有效，因为不同时间步的数据分布差异很大
3. 提供了一种注入条件信息的强大机制

常见变体：
- AdaIN (Adaptive Instance Norm): 风格迁移
- AdaLN (Adaptive Layer Norm): Transformer 扩散模型
- AdaGN (Adaptive Group Norm): UNet 扩散模型
"""
