import torch
from typing import List, Optional, Union

"""
Stable Diffusion 文生图管道

核心组件：
- vae: 图像编解码器（AutoencoderKL）
- text_encoder: CLIP 文本编码器
- tokenizer: CLIP 分词器
- unet: 条件 UNet，执行去噪
- scheduler: 控制去噪过程
"""

class StableDiffusionPipeline:
    """
    Stable Diffusion 文本到图像生成管道

    核心流程：
    1. 文本编码：prompt -> CLIP text encoder -> text embeddings
    2. 随机噪声初始化
    3. 迭代去噪：UNet 在 text embeddings 条件下预测并移除噪声
    4. VAE 解码：latent -> image
    """

    def __init__(
        self,
        vae,                # AutoencoderKL
        text_encoder,       # CLIPTextModel
        tokenizer,          # CLIPTokenizer
        unet,               # UNet2DConditionModel
        scheduler,          # 如 DDIMScheduler, EulerDiscreteScheduler
        safety_checker=None,
        feature_extractor=None,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.safety_checker = safety_checker
        self.feature_extractor = feature_extractor

        # VAE 下采样倍数（通常是8，即512->64）
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    def encode_prompt(self, prompt, device, num_images_per_prompt=1, do_classifier_free_guidance=True):
        """
        将文本 prompt 编码为 text embeddings

        参数:
            prompt: 单个字符串或字符串列表
            do_classifier_free_guidance: 是否使用 CFG（需要同时编码空 prompt）
        """
        # 1. 分词
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
        text_input_ids = text_inputs.input_ids.to(device)

        # 2. 编码
        prompt_embeds = self.text_encoder(text_input_ids)[0]

        # 3. 复制以支持每个 prompt 生成多张图
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1, prompt_embeds.shape[-1])

        # 4. 如果使用 CFG，还需要编码空 prompt（无条件）
        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device))[0]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, -1, -1)

            # 拼接：[negative, positive]
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,      # CFG 强度，>1 启用
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,                 # DDIM 参数
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
    ):
        """
        生成图像

        参数:
            prompt: 文本提示词
            height/width: 生成图像尺寸（必须是 8 的倍数）
            num_inference_steps: 去噪步数
            guidance_scale: CFG 引导强度，建议 7.5
            negative_prompt: 负面提示词

        示例:
            pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
            image = pipe("a photo of an astronaut riding a horse on mars").images[0]
        """
        device = self.unet.device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. 编码 prompt
        prompt_embeds = self.encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance
        )

        # 2. 准备时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. 准备 latent 空间的初始噪声
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        num_channels_latents = self.unet.config.in_channels
        latents_shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = torch.randn(latents_shape, generator=generator, device=device, dtype=prompt_embeds.dtype)

        # 4. 缩放初始噪声（某些调度器需要）
        latents = latents * self.scheduler.init_noise_sigma

        # 5. 去噪循环
        for i, t in enumerate(timesteps):
            # 如果使用 CFG，需要对 latent 做两次前向传播
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 预测噪声残差
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            # 应用 Classifier-Free Guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 计算前一个时间步的 latent：x_t -> x_{t-1}
            latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample

        # 6. VAE 解码
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample

        # 7. 后处理
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # 8. 可选：安全检查
        if self.safety_checker is not None:
            # 简化版不实现，实际使用时会检测 NSFW 内容
            pass

        # 9. 转换为 PIL
        if output_type == "pil":
            from ..pipeline_utils import numpy_to_pil
            image = numpy_to_pil(image)

        return image


# ============================================================================
# 核心概念说明
# ============================================================================
"""
Stable Diffusion 的核心技术：

1. Latent Diffusion (潜在扩散)
   - 不在像素空间操作，而是在 VAE 的 latent 空间（通常是 64x64）
   - 大大降低计算成本（相比 512x512 像素空间）
   - VAE 将 512x512 图像压缩为 64x64 latent

2. Classifier-Free Guidance (CFG)
   - 同时使用有条件和无条件预测
   - noise_final = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
   - guidance_scale > 1：增强文本引导，生成更符合 prompt 的图像
   - guidance_scale = 1：无引导，等同于普通条件生成
   - 推荐值：7.5

3. CLIP Text Encoder
   - 使用预训练的 CLIP 模型编码文本
   - 输出 77x768 的 text embeddings
   - 通过 cross-attention 注入到 UNet 中

4. UNet 架构
   - 编码器：逐步下采样
   - 瓶颈层：最低分辨率的特征
   - 解码器：逐步上采样，带跳跃连接
   - Cross-Attention：在每层注入 text embeddings

5. VAE (Variational Autoencoder)
   - Encoder: image (512x512) -> latent (64x64)
   - Decoder: latent (64x64) -> image (512x512)
   - 下采样倍数: 8 (= 2^3)
"""
