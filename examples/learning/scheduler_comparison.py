"""
调度器对比学习 - 理解不同调度器的差异
==========================================

这个示例对比了几种经典的扩散模型调度器：
1. DDPM - 原始调度器
2. DDIM - 确定性/加速采样
3. Euler - 高效的ODE求解器
4. SDE - 基于随机微分方程

运行要求：
- pip install diffusers torch matplotlib

作者：Diffusers 精简学习版
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)


def compare_beta_schedules():
    """
    对比不同的 beta 调度策略
    beta 决定了每一步添加多少噪声
    """
    print("=" * 60)
    print("对比1: Beta 调度策略")
    print("=" * 60)

    num_timesteps = 1000

    # 1. 线性调度
    linear_betas = torch.linspace(0.0001, 0.02, num_timesteps)

    # 2. 缩放线性调度 (用于潜在扩散)
    scaled_linear_betas = torch.linspace(0.0001**0.5, 0.02**0.5, num_timesteps) ** 2

    # 3. 余弦调度 (Improved DDPM)
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    cosine_betas = cosine_beta_schedule(num_timesteps)

    print("\n不同 beta 调度的特点:")
    print("  - 线性调度: 简单直接，beta 均匀增长")
    print("  - 缩放线性: 在低维潜在空间更稳定")
    print("  - 余弦调度: 噪声变化更平滑，生成质量更高")

    print("\n各调度的 beta 范围:")
    print(f"  线性:     [{linear_betas[0]:.6f}, {linear_betas[-1]:.6f}]")
    print(f"  缩放线性: [{scaled_linear_betas[0]:.6f}, {scaled_linear_betas[-1]:.6f}]")
    print(f"  余弦:     [{cosine_betas[0]:.6f}, {cosine_betas[-1]:.6f}]")


def compare_ddpm_vs_ddim():
    """
    对比 DDPM 和 DDIM 的差异
    """
    print("\n" + "=" * 60)
    print("对比2: DDPM vs DDIM")
    print("=" * 60)

    comparison = """
┌─────────────────────────────────────────────────────────────┐
│                   DDPM vs DDIM 对比                         │
├───────────────────┬───────────────────────┬─────────────────┤
│     特性          │       DDPM            │      DDIM       │
├───────────────────┼───────────────────────┼─────────────────┤
│ 采样确定性        │ 随机（每次不同）      │ 可确定（eta=0） │
│ 推理步数          │ 需要完整步数（1000）  │ 可跳步（50步）  │
│ 采样公式          │ 包含随机噪声项        │ 可无随机项      │
│ 语义插值          │ 不支持                │ 支持            │
│ 生成速度          │ 慢                    │ 快              │
│ 生成多样性        │ 高                    │ 可控            │
└───────────────────┴───────────────────────┴─────────────────┘

DDPM 采样公式:
  x_{t-1} = μ_θ(x_t, t) + σ_t * z,  z ~ N(0, I)

DDIM 采样公式:
  x_{t-1} = √(ᾱ_{t-1}) * pred_x0 + √(1-ᾱ_{t-1}-σ²) * ε_θ + σ * z

  当 eta=0 时，σ=0，采样变为确定性的！
"""
    print(comparison)

    # 创建调度器进行对比
    ddpm = DDPMScheduler(num_train_timesteps=1000)
    ddim = DDIMScheduler(num_train_timesteps=1000)

    # 设置推理步数
    ddpm.set_timesteps(1000)  # DDPM 通常需要完整步数
    ddim.set_timesteps(50)    # DDIM 可以大幅减少步数

    print(f"\n实际对比:")
    print(f"  DDPM 推理步数: {len(ddpm.timesteps)}")
    print(f"  DDIM 推理步数: {len(ddim.timesteps)}")
    print(f"  速度提升: {len(ddpm.timesteps) / len(ddim.timesteps):.1f}x")


def explain_eta_parameter():
    """
    解释 DDIM 的 eta 参数
    """
    print("\n" + "=" * 60)
    print("深入理解: DDIM 的 eta 参数")
    print("=" * 60)

    explanation = """
DDIM 中的 eta (η) 参数控制采样的随机性:

公式: σ_t = η * √((1-ᾱ_{t-1})/(1-ᾱ_t)) * √(1-ᾱ_t/ᾱ_{t-1})

┌─────────────────────────────────────────────────────────┐
│     eta 值      │            效果                       │
├─────────────────┼───────────────────────────────────────┤
│    eta = 0      │  确定性采样，相同噪声→相同结果        │
│    eta = 0.5    │  中等随机性                           │
│    eta = 1.0    │  完全随机，等价于 DDPM                │
└─────────────────┴───────────────────────────────────────┘

使用场景:
- eta=0: 适合需要可复现结果的场景
- eta=0.5-0.8: 平衡质量和多样性
- eta=1: 需要最大多样性时
"""
    print(explanation)

    # 演示不同 eta 的效果
    ddim = DDIMScheduler(num_train_timesteps=1000)
    ddim.set_timesteps(50)

    print("\n模拟相同初始噪声下不同 eta 的结果:")
    torch.manual_seed(42)

    for eta in [0.0, 0.5, 1.0]:
        # 模拟采样（简化演示）
        print(f"  eta={eta}: {'确定性' if eta == 0 else '随机性'} 采样")


def compare_euler_schedulers():
    """
    对比 Euler 系列调度器
    """
    print("\n" + "=" * 60)
    print("对比3: Euler 调度器家族")
    print("=" * 60)

    comparison = """
Euler 调度器基于 ODE/SDE 数值求解器:

┌──────────────────────────────────────────────────────────────┐
│        调度器               │           特点                 │
├─────────────────────────────┼────────────────────────────────┤
│ EulerDiscreteScheduler      │ 一阶 ODE 求解器，快速稳定      │
│ EulerAncestralScheduler     │ 带祖先采样，增加随机性         │
└─────────────────────────────┴────────────────────────────────┘

Euler 方法原理:
  dx/dt = f(x, t)
  x_{t+Δt} ≈ x_t + Δt * f(x_t, t)

在扩散模型中:
  f(x_t, t) = -σ'(t)/σ(t) * (x_t - D(x_t; σ(t)))

其中 D 是去噪网络，σ(t) 是噪声水平
"""
    print(comparison)

    # 创建调度器
    euler = EulerDiscreteScheduler(num_train_timesteps=1000)
    euler_a = EulerAncestralDiscreteScheduler(num_train_timesteps=1000)

    euler.set_timesteps(20)  # Euler 可以用很少的步数
    euler_a.set_timesteps(20)

    print(f"\n推荐步数:")
    print(f"  Euler: 20-50 步即可获得良好结果")
    print(f"  Euler Ancestral: 20-50 步，多样性更高")


def summary_table():
    """
    总结表格
    """
    print("\n" + "=" * 60)
    print("总结: 如何选择调度器")
    print("=" * 60)

    summary = """
┌───────────────────────────────────────────────────────────────┐
│                 调度器选择指南                                 │
├───────────────────┬───────────────────────────────────────────┤
│      需求          │           推荐调度器                       │
├───────────────────┼───────────────────────────────────────────┤
│ 学习/理解原理      │ DDPMScheduler                             │
│ 快速生成          │ DDIMScheduler (20-50步)                    │
│ 最高质量          │ EulerDiscreteScheduler                     │
│ 最大多样性        │ EulerAncestralDiscreteScheduler            │
│ 确定性结果        │ DDIMScheduler (eta=0)                      │
│ 科学研究          │ ScoreSdeVeScheduler                        │
└───────────────────┴───────────────────────────────────────────┘

常用配置:
1. Stable Diffusion 默认: EulerDiscreteScheduler, 20-30步
2. 高质量生成: DDIM, 50步, eta=0
3. 快速预览: Euler, 10-20步
"""
    print(summary)


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("           调度器对比学习")
    print("=" * 70)

    compare_beta_schedules()
    compare_ddpm_vs_ddim()
    explain_eta_parameter()
    compare_euler_schedulers()
    summary_table()

    print("\n" + "=" * 70)
    print("           学习完成！")
    print("=" * 70)
    print("\n下一步:")
    print("1. 阅读各调度器源码，理解公式实现")
    print("2. 使用真实模型测试不同调度器效果")
    print("3. 调整步数和参数，观察质量变化")


if __name__ == "__main__":
    main()
