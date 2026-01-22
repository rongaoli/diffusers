import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

"""
DDIM (Denoising Diffusion Implicit Models) 调度器

论文: https://arxiv.org/abs/2010.02502

核心优势：
- 确定性采样（非马尔可夫过程）
- 可以大幅减少采样步数（50步甚至10步）
- 通过 eta 参数控制随机性
"""

@dataclass
class DDIMSchedulerOutput:
    """调度器 step 函数的输出"""
    prev_sample: torch.Tensor          # x_{t-1}，前一个时间步的样本
    pred_original_sample: Optional[torch.Tensor] = None  # x_0 预测


class DDIMScheduler:
    """
    DDIM 调度器

    核心思想：
    - DDPM 是马尔可夫过程（每步只依赖前一步）
    - DDIM 是非马尔可夫的，可以跳步采样
    - 通过重参数化实现确定性/随机性控制
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,     # 训练时的总步数
        beta_start: float = 0.0001,          # 噪声调度起始值
        beta_end: float = 0.02,              # 噪声调度结束值
        beta_schedule: str = "linear",       # 噪声调度类型
        clip_sample: bool = True,            # 是否裁剪预测的 x_0
        set_alpha_to_one: bool = True,       # 最后一步是否设置 alpha=1
        prediction_type: str = "epsilon",    # 预测类型：epsilon/v_prediction
    ):
        # 1. 计算 beta 调度
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # Stable Diffusion 使用的调度
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        else:
            raise ValueError(f"未知的 beta_schedule: {beta_schedule}")

        # 2. 计算 alpha 相关参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # α_t = ∏(1-β_i)

        # 为了计算方便，补充一个 alpha_0 = 1
        if set_alpha_to_one:
            self.final_alpha_cumprod = torch.tensor(1.0)
        else:
            self.final_alpha_cumprod = self.alphas_cumprod[0]

        # 配置
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type

        # 推理时的时间步（通过 set_timesteps 设置）
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def set_timesteps(self, num_inference_steps: int, device: str = None):
        """
        设置推理时的时间步

        DDIM 的关键优势：可以使用比训练时更少的步数
        例如：训练用 1000 步，推理只用 50 步
        """
        # 均匀采样时间步
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.num_inference_steps = num_inference_steps

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """缩放模型输入（DDIM 不需要缩放）"""
        return sample

    def step(
        self,
        model_output: torch.Tensor,      # 模型预测的噪声 ε_θ
        timestep: int,                    # 当前时间步 t
        sample: torch.Tensor,             # 当前样本 x_t
        eta: float = 0.0,                 # 随机性控制：0=确定性，1=DDPM
        generator: Optional[torch.Generator] = None,
    ) -> DDIMSchedulerOutput:
        """
        DDIM 采样步骤：x_t -> x_{t-1}

        公式：
        x_{t-1} = √(α_{t-1}) * x_0_pred + √(1 - α_{t-1} - σ_t^2) * ε_t + σ_t * ε

        其中：
        - x_0_pred: 从 x_t 和 ε_θ 预测的原始图像
        - σ_t = eta * √((1-α_{t-1})/(1-α_t)) * √(1-α_t/α_{t-1})
        - ε: 随机噪声（eta=0 时不加）
        """
        # 1. 获取当前和前一个时间步的参数
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 2. 根据 model_output 预测原始图像 x_0
        if self.prediction_type == "epsilon":
            # 模型预测的是噪声 ε
            # x_0 = (x_t - √(1-α_t) * ε) / √α_t
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.prediction_type == "v_prediction":
            # 模型预测的是 v = √α_t * ε - √(1-α_t) * x_0
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        else:
            raise ValueError(f"未知的 prediction_type: {self.prediction_type}")

        # 3. 裁剪 x_0 预测（防止数值不稳定）
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1, 1)

        # 4. 计算方差 σ_t^2
        # DDIM 论文公式：σ_t = eta * √((1-α_{t-1})/(1-α_t) * (1-α_t/α_{t-1}))
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** 0.5

        # 5. 计算 "方向向量" 指向 x_t
        # pred_sample_direction = √(1-α_{t-1}-σ_t^2) * ε
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * model_output

        # 6. 组合得到 x_{t-1}
        # x_{t-1} = √α_{t-1} * x_0 + pred_sample_direction + σ_t * noise
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        # 7. 添加随机噪声（如果 eta > 0）
        if eta > 0:
            noise = torch.randn(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
            prev_sample = prev_sample + std_dev_t * noise

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def _get_variance(self, timestep, prev_timestep):
        """计算方差"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        给干净图像添加噪声（用于训练或 img2img）

        前向扩散公式：
        x_t = √α_t * x_0 + √(1-α_t) * ε
        """
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    @property
    def init_noise_sigma(self):
        """初始噪声的缩放系数（某些调度器需要）"""
        return 1.0


# ============================================================================
# 核心概念说明
# ============================================================================
"""
DDIM vs DDPM 的核心区别：

1. 采样过程
   DDPM (马尔可夫):
   - x_{t-1} 只依赖 x_t
   - 必须遍历所有时间步
   - 1000 步训练 -> 1000 步推理

   DDIM (非马尔可夫):
   - 可以跳步采样
   - 1000 步训练 -> 50 步推理
   - 大幅加速！

2. 确定性 vs 随机性
   DDPM:
   - 总是随机的（每步加噪声）
   - 相同输入产生不同输出

   DDIM (eta=0):
   - 完全确定性
   - 相同输入产生相同输出
   - 方便图像编辑、插值

3. Eta 参数的作用
   eta = 0: 完全确定性（DDIM）
   eta = 1: 等同于 DDPM
   0 < eta < 1: 插值

4. 采样质量
   - 步数少时（<50）: DDIM 通常更好
   - 步数多时（>100）: DDPM 和 DDIM 相当
   - 实践中常用 DDIM 50步

5. 推理加速示例
   设置 50 步推理：
   scheduler.set_timesteps(50)
   实际时间步：[980, 960, 940, ..., 20, 0]
   跳过大部分中间步骤

6. 为什么 DDIM 可以跳步？
   DDPM: p(x_{t-1}|x_t) 是固定的高斯分布
   DDIM: 重新设计前向过程，使得跳步成为可能
   关键：构造非马尔可夫的前向过程
"""
