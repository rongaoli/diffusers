import torch
from typing import List, Optional, Union
import PIL.Image

"""
Stable Diffusion Inpainting 管道（图像修复/补全）

用途：
- 图像修复：去除不想要的物体
- 图像补全：填充缺失区域
- 局部编辑：只修改 mask 区域

关键输入：
- image: 原始图像
- mask: 指定要修改的区域（白色=修改，黑色=保留）
- prompt: 描述想要生成的内容
"""

class StableDiffusionInpaintPipeline:
    """
    Stable Diffusion Inpainting 管道

    核心思想：
    - 在去噪过程中，mask 外的区域保持不变
    - 只在 mask 内的区域生成新内容
    - 通过将 mask 和原图信息作为额外输入传给 UNet
    """

    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    def encode_prompt(self, prompt, device, do_classifier_free_guidance=True):
        """编码文本 prompt"""
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

    def prepare_mask_and_image(self, image, mask, height, width, device, dtype):
        """
        准备 mask 和图像

        步骤：
        1. 将 mask 调整为正确尺寸并归一化
        2. 将图像编码为 latent
        3. 将 mask 下采样到 latent 空间尺寸
        """
        # 处理 mask
        if isinstance(mask, PIL.Image.Image):
            mask = mask.resize((width, height), PIL.Image.LANCZOS)
            from torchvision import transforms
            mask = transforms.ToTensor()(mask.convert("L"))
            mask = mask.to(device, dtype=dtype)
        else:
            mask = mask.to(device, dtype=dtype)

        # mask 值归一化到 [0, 1]，其中 1=要修复的区域
        mask = mask.mean(dim=0, keepdim=True) if mask.ndim == 3 else mask

        # 处理图像
        if isinstance(image, PIL.Image.Image):
            image = image.resize((width, height), PIL.Image.LANCZOS)
            from torchvision import transforms
            image = transforms.ToTensor()(image)
            image = image.to(device, dtype=dtype)
            image = 2.0 * image - 1.0
        else:
            image = image.to(device, dtype=dtype)

        # 编码图像为 latent
        image_latents = self.vae.encode(image.unsqueeze(0)).latent_dist.sample()
        image_latents = self.vae.config.scaling_factor * image_latents

        # 将 mask 下采样到 latent 尺寸
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0),
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
        )

        return mask, image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[PIL.Image.Image, torch.Tensor],
        mask_image: Union[PIL.Image.Image, torch.Tensor],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
    ):
        """
        Inpainting 图像生成

        参数:
            prompt: 文本提示词（描述要生成的内容）
            image: 原始图像
            mask_image: 遮罩图像（白色=修改区域，黑色=保留区域）

        示例:
            pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")

            # 加载图像和遮罩
            init_image = PIL.Image.open("dog.png")
            mask_image = PIL.Image.open("mask.png")  # 白色区域将被重绘

            prompt = "a cat, high quality, detailed"
            image = pipe(prompt, init_image, mask_image).images[0]
        """
        device = self.unet.device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. 编码 prompt
        prompt_embeds = self.encode_prompt(prompt, device, do_classifier_free_guidance)

        # 2. 准备 mask 和图像
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        mask, masked_image_latents = self.prepare_mask_and_image(
            image, mask_image, height, width, device, prompt_embeds.dtype
        )

        # 3. 准备时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. 准备初始噪声
        num_channels_latents = self.unet.config.in_channels
        latents_shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        # 5. 去噪循环
        for t in timesteps:
            # 关键：将 mask 和 masked_image_latents 拼接到 latents
            # UNet 输入：[latents, mask, masked_image_latents]
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # 拼接 mask 和 masked_image（作为额外的条件信息）
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 预测噪声
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            # CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 去噪一步
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
Inpainting (图像修复) 的核心原理：

1. 与 img2img 的区别
   - img2img: 修改整张图像
   - inpainting: 只修改 mask 标记的区域

2. 输入格式
   - image: 原始图像
   - mask: 二值图像
     * 白色(255): 需要修复/重绘的区域
     * 黑色(0): 保持不变的区域
   - prompt: 描述要生成什么内容

3. UNet 输入通道
   标准 SD: 4 个通道 (latent)
   Inpainting SD: 9 个通道
   - 4: 当前去噪的 latent
   - 1: mask (下采样到 latent 尺寸)
   - 4: masked_image_latent (被遮挡图像的 latent)

4. 训练与推理
   训练时：
   - 模型学会根据 mask 和周围信息填充缺失区域
   - 使用大量 image + random mask + inpainted result 数据

   推理时：
   - 提供原图、mask、prompt
   - 模型只重绘 mask 区域
   - mask 外的区域通过条件输入保持一致

5. 典型应用
   - 物体移除：mask 覆盖不想要的物体，prompt 描述背景
   - 物体替换：mask 覆盖物体，prompt 描述新物体
   - 图像扩展：mask 覆盖边缘，prompt 描述扩展内容
   - 人脸修复：mask 覆盖缺陷，prompt 描述修复目标

6. 创建 Mask 的方法
   - 手动绘制（Photoshop, GIMP）
   - 自动分割（SAM, Grounded-SAM）
   - 简单阈值（基于颜色）

7. 注意事项
   - 需要使用专门的 inpainting 模型（如 stable-diffusion-inpainting）
   - 标准 SD 模型的 UNet 只有 4 个输入通道，不支持 inpainting
   - Mask 边缘可能有接缝，可以使用 feathering（羽化）改善
"""
