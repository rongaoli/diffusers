# Diffusers 精简学习版 - 中文指南

这是 HuggingFace Diffusers 库的精简版本，专门为学习和理解扩散模型（Diffusion Models）而设计。

## 目录

1. [扩散模型基础概念](#扩散模型基础概念)
2. [代码结构](#代码结构)
3. [核心组件详解](#核心组件详解)
4. [快速开始](#快速开始)
5. [学习路径](#学习路径)

---

## 扩散模型基础概念

### 什么是扩散模型？

扩散模型是一类强大的生成模型，其核心思想基于两个过程：

#### 1. 前向扩散过程（Forward Diffusion）
逐步向数据添加高斯噪声，直到数据变成纯随机噪声：

```
x_0 → x_1 → x_2 → ... → x_T
原始图像               纯噪声

每一步添加少量噪声：
x_t = √(α_t) * x_{t-1} + √(1-α_t) * ε
其中 ε ~ N(0, I) 是标准高斯噪声
```

#### 2. 反向去噪过程（Reverse Denoising）
训练神经网络学习去噪，从纯噪声逐步恢复原始数据：

```
x_T → x_{T-1} → ... → x_1 → x_0
纯噪声                    生成的图像

神经网络预测噪声 ε_θ(x_t, t)，然后去除噪声
```

### 关键公式

**DDPM 采样公式：**
```
x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z
```

其中：
- `α_t = 1 - β_t` 是保留原始信号的比例
- `ᾱ_t = ∏_{s=1}^t α_s` 是累积保留比例
- `β_t` 是噪声调度参数
- `ε_θ` 是神经网络预测的噪声
- `z ~ N(0, I)` 是随机噪声（用于采样随机性）

---

## 代码结构

```
src/diffusers/
├── __init__.py              # 主入口文件
├── configuration_utils.py   # 配置基类
├── image_processor.py       # 图像处理工具
│
├── models/                  # 神经网络模型
│   ├── unets/              # UNet 模型（扩散模型骨干网络）
│   │   ├── unet_2d.py              # 基础 2D UNet
│   │   ├── unet_2d_condition.py    # 条件 2D UNet
│   │   └── unet_2d_blocks.py       # UNet 构建块
│   │
│   ├── autoencoders/       # 自编码器（用于潜在扩散）
│   │   ├── autoencoder_kl.py       # KL-VAE
│   │   └── vae.py                  # VAE 基础组件
│   │
│   ├── transformers/       # Transformer 模型
│   │   ├── transformer_2d.py       # 通用 Transformer
│   │   ├── dit_transformer_2d.py   # DiT 模型
│   │   └── prior_transformer.py    # 先验 Transformer
│   │
│   ├── attention.py         # 注意力机制核心
│   ├── attention_processor.py # 注意力处理器
│   ├── embeddings.py        # 嵌入层（时间步、位置等）
│   ├── normalization.py     # 归一化层
│   └── resnet.py            # ResNet 块
│
├── schedulers/              # 噪声调度器
│   ├── scheduling_ddpm.py           # DDPM 调度器
│   ├── scheduling_ddim.py           # DDIM 调度器
│   ├── scheduling_euler_discrete.py # Euler 调度器
│   ├── scheduling_sde_ve.py         # Score SDE 调度器
│   ├── scheduling_pndm.py           # PNDM 调度器
│   └── scheduling_utils.py          # 调度器基类
│
├── pipelines/               # 推理管道
│   ├── ddpm/               # DDPM 管道
│   │   └── pipeline_ddpm.py
│   │
│   ├── ddim/               # DDIM 管道
│   │   └── pipeline_ddim.py
│   │
│   ├── dit/                # DiT 管道
│   │   └── pipeline_dit.py
│   │
│   ├── latent_diffusion/   # 潜在扩散管道
│   │   ├── pipeline_latent_diffusion.py
│   │   └── pipeline_latent_diffusion_superresolution.py
│   │
│   ├── stable_diffusion/   # Stable Diffusion 管道
│   │   ├── pipeline_stable_diffusion.py
│   │   ├── pipeline_stable_diffusion_img2img.py
│   │   └── pipeline_stable_diffusion_inpaint.py
│   │
│   └── pipeline_utils.py   # 管道基类
│
└── utils/                   # 工具函数
```

---

## 核心组件详解

### 1. 调度器（Scheduler）

调度器控制扩散过程中的噪声添加和去除策略。

#### DDPMScheduler（去噪扩散概率模型调度器）

```python
# 文件: schedulers/scheduling_ddpm.py

class DDPMScheduler:
    """
    DDPM 调度器 - 最基础的扩散调度器

    核心参数:
    - num_train_timesteps: 总时间步数（通常1000）
    - beta_start/beta_end: 噪声调度的起止值
    - beta_schedule: 噪声调度类型（linear, cosine等）

    核心方法:
    - add_noise(): 前向过程，添加噪声
    - step(): 反向过程，去除噪声
    """

    def add_noise(self, original_samples, noise, timesteps):
        """
        前向扩散：给原始样本添加噪声

        公式: x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε

        参数:
        - original_samples: 原始干净样本 x_0
        - noise: 随机噪声 ε
        - timesteps: 时间步 t

        返回: 加噪后的样本 x_t
        """
        pass

    def step(self, model_output, timestep, sample):
        """
        反向去噪：根据模型预测去除噪声

        公式: x_{t-1} = (x_t - β_t/√(1-ᾱ_t) * ε_θ) / √α_t + σ_t * z

        参数:
        - model_output: 神经网络预测的噪声 ε_θ
        - timestep: 当前时间步 t
        - sample: 当前噪声样本 x_t

        返回: 去噪后的样本 x_{t-1}
        """
        pass
```

#### DDIMScheduler（去噪扩散隐式模型调度器）

```python
# 文件: schedulers/scheduling_ddim.py

class DDIMScheduler:
    """
    DDIM 调度器 - DDPM 的改进版本

    优势:
    1. 确定性采样（给定相同噪声，生成相同结果）
    2. 可跳步采样（用更少步数完成生成）
    3. 支持语义插值

    核心参数:
    - eta: 控制采样随机性（0=确定性，1=完全随机）
    """

    def step(self, model_output, timestep, sample, eta=0.0):
        """
        DDIM 采样步骤

        公式（简化）:
        x_{t-1} = √(ᾱ_{t-1}) * pred_x0 + √(1-ᾱ_{t-1}-σ²) * ε_θ + σ * noise

        其中 pred_x0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        """
        pass
```

### 2. 模型（Model）

#### UNet2DModel（基础UNet）

```python
# 文件: models/unets/unet_2d.py

class UNet2DModel:
    """
    2D UNet 模型 - 扩散模型的骨干网络

    架构特点:
    1. 编码器-解码器结构
    2. 跳跃连接（Skip Connections）
    3. 下采样和上采样块
    4. 时间步嵌入

    输入: (batch, channels, height, width), timestep
    输出: (batch, channels, height, width) - 预测的噪声
    """
    pass
```

#### UNet2DConditionModel（条件UNet）

```python
# 文件: models/unets/unet_2d_condition.py

class UNet2DConditionModel:
    """
    条件 2D UNet - 支持文本等条件输入

    额外功能:
    1. 交叉注意力层（Cross-Attention）
    2. 条件嵌入（如文本嵌入）
    3. 用于文本到图像生成

    输入:
    - sample: 噪声图像
    - timestep: 时间步
    - encoder_hidden_states: 条件嵌入（如CLIP文本嵌入）

    输出: 预测的噪声
    """
    pass
```

#### AutoencoderKL（变分自编码器）

```python
# 文件: models/autoencoders/autoencoder_kl.py

class AutoencoderKL:
    """
    KL-VAE - 用于潜在扩散模型

    功能:
    1. 将图像编码到低维潜在空间
    2. 将潜在表示解码回图像
    3. 大幅降低扩散的计算成本

    编码: (B, 3, H, W) -> (B, 4, H/8, W/8)
    解码: (B, 4, H/8, W/8) -> (B, 3, H, W)

    在潜在空间进行扩散，效率提升约64倍！
    """

    def encode(self, x):
        """将图像编码到潜在空间"""
        pass

    def decode(self, z):
        """将潜在表示解码为图像"""
        pass
```

### 3. 管道（Pipeline）

#### DDPMPipeline

```python
# 文件: pipelines/ddpm/pipeline_ddpm.py

class DDPMPipeline:
    """
    DDPM 管道 - 最基础的扩散生成管道

    组件:
    - unet: UNet2DModel
    - scheduler: DDPMScheduler

    生成流程:
    1. 从纯噪声开始 x_T ~ N(0, I)
    2. 循环 T 次去噪:
       a. 预测噪声: ε_θ = unet(x_t, t)
       b. 去噪一步: x_{t-1} = scheduler.step(ε_θ, t, x_t)
    3. 返回生成的图像 x_0
    """

    def __call__(self, batch_size=1, num_inference_steps=1000):
        # 1. 采样初始噪声
        image = torch.randn((batch_size, 3, 256, 256))

        # 2. 设置时间步
        self.scheduler.set_timesteps(num_inference_steps)

        # 3. 去噪循环
        for t in self.scheduler.timesteps:
            # 预测噪声
            noise_pred = self.unet(image, t).sample
            # 去噪一步
            image = self.scheduler.step(noise_pred, t, image).prev_sample

        # 4. 返回结果
        return image
```

#### StableDiffusionPipeline

```python
# 文件: pipelines/stable_diffusion/pipeline_stable_diffusion.py

class StableDiffusionPipeline:
    """
    Stable Diffusion 管道 - 文本到图像生成

    组件:
    - vae: AutoencoderKL（VAE编码器/解码器）
    - text_encoder: CLIPTextModel（文本编码器）
    - tokenizer: CLIPTokenizer（分词器）
    - unet: UNet2DConditionModel（条件UNet）
    - scheduler: 调度器

    生成流程:
    1. 文本编码: prompt -> CLIP -> text_embeddings
    2. 在潜在空间采样噪声: latents ~ N(0, I)
    3. 条件去噪循环:
       a. 噪声预测: ε_θ = unet(latents, t, text_embeddings)
       b. 去噪: latents = scheduler.step(ε_θ, t, latents)
    4. 解码: image = vae.decode(latents)
    """

    def __call__(self, prompt, num_inference_steps=50):
        # 1. 文本编码
        text_embeddings = self.encode_prompt(prompt)

        # 2. 采样潜在噪声
        latents = torch.randn((1, 4, 64, 64))  # 512x512 图像的潜在表示

        # 3. 去噪循环（带条件）
        for t in self.scheduler.timesteps:
            # 预测噪声（使用文本条件）
            noise_pred = self.unet(latents, t, text_embeddings).sample
            # 去噪
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 4. 解码到图像空间
        image = self.vae.decode(latents)

        return image
```

---

## 快速开始

### 基础示例：DDPM 图像生成

```python
from diffusers import DDPMPipeline
import torch

# 加载预训练模型
pipeline = DDPMPipeline.from_pretrained("google/ddpm-cat-256")
pipeline.to("cuda")

# 生成图像
with torch.no_grad():
    images = pipeline(batch_size=4).images

# 保存图像
for i, img in enumerate(images):
    img.save(f"cat_{i}.png")
```

### 进阶示例：Stable Diffusion 文本到图像

```python
from diffusers import StableDiffusionPipeline
import torch

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# 生成图像
prompt = "一只可爱的猫咪在阳光下睡觉，高清摄影"
image = pipe(prompt, num_inference_steps=50).images[0]
image.save("cat_sleeping.png")
```

---

## 学习路径

### 第一阶段：理解基础
1. 阅读 `schedulers/scheduling_ddpm.py` - 理解噪声调度
2. 阅读 `pipelines/ddpm/pipeline_ddpm.py` - 理解生成流程
3. 运行 DDPM 示例，观察生成过程

### 第二阶段：深入模型
1. 阅读 `models/unets/unet_2d.py` - 理解 UNet 架构
2. 阅读 `models/attention.py` - 理解注意力机制
3. 修改模型参数，观察效果变化

### 第三阶段：潜在扩散
1. 阅读 `models/autoencoders/autoencoder_kl.py` - 理解 VAE
2. 阅读 `pipelines/stable_diffusion/` - 理解完整流程
3. 尝试不同的提示词和参数

### 第四阶段：进阶优化
1. 阅读 `schedulers/scheduling_ddim.py` - 理解加速采样
2. 比较不同调度器的效果
3. 尝试训练自己的小型模型

---

## 推荐论文

1. **DDPM**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
   - https://arxiv.org/abs/2006.11239
   - 扩散模型的基础论文

2. **DDIM**: "Denoising Diffusion Implicit Models" (Song et al., 2020)
   - https://arxiv.org/abs/2010.02502
   - 加速采样方法

3. **LDM**: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
   - https://arxiv.org/abs/2112.10752
   - Stable Diffusion 的基础

4. **DiT**: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2022)
   - https://arxiv.org/abs/2212.09748
   - Transformer 在扩散模型中的应用

5. **Score SDE**: "Score-Based Generative Modeling through Stochastic Differential Equations" (Song et al., 2020)
   - https://arxiv.org/abs/2011.13456
   - 统一的 SDE 视角

---

## 常见问题

### Q: 为什么要使用潜在空间扩散？
A: 图像是高维数据（如512x512x3），直接在像素空间扩散计算量巨大。VAE将图像压缩到低维潜在空间（如64x64x4），计算量降低约64倍，同时保持生成质量。

### Q: DDPM 和 DDIM 的区别？
A: DDPM每步采样有随机噪声，结果不确定；DDIM可以设置eta=0实现确定性采样，且可以跳步加速（如50步而非1000步）。

### Q: 什么是 Classifier-Free Guidance？
A: 一种提升条件生成质量的技术。同时预测有条件和无条件的噪声，然后放大条件方向：
`ε = ε_uncond + guidance_scale * (ε_cond - ε_uncond)`

---

祝学习愉快！如有问题，欢迎查阅源代码注释。
