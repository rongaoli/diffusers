import torch
from typing import List, Optional, Union

class DiTPipeline:
    """
    基于 Transformer 的图像生成管道（而非传统 UNet）

    核心组件：
    - transformer: DiT模型，用于去噪
    - vae: 编码/解码图像
    - scheduler: 控制去噪步骤
    """

    def __init__(self, transformer, vae, scheduler, id2label=None):
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler

        # 构建 ImageNet 标签字典（标签名 -> 类别ID）
        self.labels = {}
        if id2label:
            for key, value in id2label.items():
                for label in value.split(","):
                    self.labels[label.strip()] = int(key)

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        """将标签名转换为类别ID"""
        if not isinstance(label, list):
            label = [label]
        return [self.labels[l] for l in label]

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],      # 要生成的图像类别
        guidance_scale: float = 4.0,   # 引导强度
        generator = None,              # 随机数生成器
        num_inference_steps: int = 50, # 去噪步数
        output_type: str = "pil",
    ):
        """
        生成图像的主函数

        示例:
            pipe = DiTPipeline(transformer, vae, scheduler)
            class_ids = pipe.get_label_ids(["white shark", "umbrella"])
            images = pipe(class_labels=class_ids, num_inference_steps=25)
        """
        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        # 1. 初始化随机噪声
        latents = torch.randn(
            batch_size, latent_channels, latent_size, latent_size,
            generator=generator,
            device=self.transformer.device,
            dtype=self.transformer.dtype
        )

        # 2. 准备条件（Classifier-Free Guidance）
        class_labels = torch.tensor(class_labels, device=latents.device)
        if guidance_scale > 1:
            # 拼接有条件和无条件（null类别=1000）
            class_null = torch.tensor([1000] * batch_size, device=latents.device)
            class_labels_input = torch.cat([class_labels, class_null])
            latents = torch.cat([latents, latents])
        else:
            class_labels_input = class_labels

        # 3. 迭代去噪
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            # 确保batch维度正确
            if guidance_scale > 1:
                half = latents[:len(latents) // 2]
                latents = torch.cat([half, half])

            # 预测噪声
            noise_pred = self.transformer(
                latents,
                timestep=t,
                class_labels=class_labels_input
            ).sample

            # 应用 Classifier-Free Guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = eps.chunk(2)
                eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([eps, eps])
                noise_pred = torch.cat([eps, rest], dim=1)

            # 如果模型预测了 sigma，只取噪声部分
            if self.transformer.config.out_channels // 2 == latent_channels:
                noise_pred, _ = noise_pred.split(latent_channels, dim=1)

            # 去噪一步：x_t -> x_{t-1}
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 4. 解码为图像
        if guidance_scale > 1:
            latents = latents.chunk(2)[0]  # 只要条件分支

        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample

        # 5. 后处理
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            from ..pipeline_utils import numpy_to_pil
            images = numpy_to_pil(images)

        return images
