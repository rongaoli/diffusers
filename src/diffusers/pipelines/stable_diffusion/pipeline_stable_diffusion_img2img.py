import torch
from typing import List, Optional, Union
import PIL.Image

"""
Stable Diffusion 图生图管道（Image-to-Image）

与文生图的区别：
- 输入：图像 + 文本
- 不从纯噪声开始，而是从编码后的图像 + 部分噪声开始
- strength 参数控制修改程度（0=不变，1=完全重绘）
"""

class StableDiffusionImg2ImgPipeline:
    """
    Stable Diffusion 图生图管道

    核心流程：
    1. 将输入图像编码为 latent
    2. 添加噪声（强度由 strength 控制）
    3. 从 t=strength*T 开始去噪（而非从 t=T）
    4. VAE 解码为图像
    """

    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    def encode_prompt(self, prompt, device, do_classifier_free_guidance=True):
        """编码文本 prompt（与 txt2img 相同）"""
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(device))[0]

        if do_classifier_free_guidance:
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            negative_embeds = self.text_encoder(uncond_input.input_ids.to(device))[0]
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds])

        return prompt_embeds

    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        """
        将输入图像转换为 VAE latent

        步骤：
        1. 图像预处理（resize, normalize）
        2. VAE 编码
        """
        if isinstance(image, PIL.Image.Image):
            image = [image]

        # 转换为 tensor 并归一化到 [-1, 1]
        if isinstance(image[0], PIL.Image.Image):
            width, height = image[0].size
            width, height = (x - x % self.vae_scale_factor for x in (width, height))

            from torchvision import transforms
            image = [img.resize((width, height), PIL.Image.LANCZOS) for img in image]
            image = [transforms.ToTensor()(img) for img in image]
            image = torch.stack(image).to(device, dtype=dtype)
            image = 2.0 * image - 1.0
        else:
            image = image.to(device, dtype=dtype)

        # VAE 编码
        image_latents = self.vae.encode(image).latent_dist.sample(generator)
        image_latents = self.vae.config.scaling_factor * image_latents

        # 复制 batch
        if batch_size > image_latents.shape[0]:
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        return image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[PIL.Image.Image, torch.Tensor],
        strength: float = 0.8,           # 修改强度：0.0=不变, 1.0=完全重绘
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
    ):
        """
        图生图生成

        参数:
            prompt: 文本提示词
            image: 输入图像（PIL Image 或 tensor）
            strength: 修改强度（0.0-1.0）
                - 0.0: 几乎不修改
                - 0.5: 中等修改
                - 0.8: 大幅修改（推荐）
                - 1.0: 完全重绘

        示例:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
            init_image = PIL.Image.open("input.jpg")
            prompt = "a fantasy landscape, trending on artstation"
            image = pipe(prompt, init_image, strength=0.75).images[0]
        """
        device = self.unet.device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. 编码 prompt
        prompt_embeds = self.encode_prompt(prompt, device, do_classifier_free_guidance)

        # 2. 准备时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 关键：根据 strength 计算起始时间步
        # strength=0.8 意味着从 80% 的噪声开始（跳过前 20% 的去噪步骤）
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]

        # 3. 准备图像 latent
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        image_latents = self.prepare_image_latents(
            image, batch_size, prompt_embeds.dtype, device, generator
        )

        # 4. 给图像 latent 添加噪声（从 t_start 对应的噪声水平开始）
        noise = torch.randn(image_latents.shape, generator=generator, device=device, dtype=image_latents.dtype)
        latents = self.scheduler.add_noise(image_latents, noise, timesteps[0:1])

        # 5. 去噪循环（从 t_start 开始，而非从头）
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 6. 解码
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            from ..pipeline_utils import numpy_to_pil
            image = numpy_to_pil(image)

        return image


# ============================================================================
# 核心概念说明
# ============================================================================
"""
图生图 (Image-to-Image) 的核心原理：

1. 与文生图的区别
   - 文生图：从纯高斯噪声开始 (t=T)
   - 图生图：从部分噪声的图像开始 (t=strength*T)

2. Strength 参数的作用
   - 控制修改程度和创造性
   - strength = 0.0: 不添加噪声，几乎保持原样
   - strength = 0.5: 中等修改，保留大部分结构
   - strength = 0.8: 大幅修改，仅保留基本构图（推荐）
   - strength = 1.0: 完全重绘，等同于文生图

3. 噪声调度
   假设总共 50 步，strength=0.8:
   - 等效噪声水平：t=40 (50 * 0.8)
   - 实际执行：从第 10 步开始（跳过前 10 步）
   - 去噪步数：40 步

4. 典型应用场景
   - 风格迁移：保留内容，改变风格
   - 图像增强：提升分辨率、改善质量
   - 局部修改：配合 inpainting mask
   - 创意变体：基于原图生成变体

5. 推荐 strength 值
   - 细微调整：0.3-0.5
   - 常规修改：0.6-0.8
   - 大幅重绘：0.8-1.0
"""
