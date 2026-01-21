# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace团队
#
# 自编码器模块 - 用于潜在扩散模型

"""
自编码器（AutoEncoder）模块
包含变分自编码器（VAE）的实现

主要模型：
- AutoencoderKL: KL散度变分自编码器

VAE在扩散模型中的作用：
1. 编码器：将图像压缩到低维潜在空间
2. 解码器：将潜在表示重建为图像
3. 潜在扩散在压缩的潜在空间进行，大幅提升效率
"""

from .autoencoder_kl import AutoencoderKL
from .vae import DiagonalGaussianDistribution, Encoder, Decoder
