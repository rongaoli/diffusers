from typing import Dict, List, Optional, Tuple, Union

import torch

from ...models import AutoencoderKL, DiTTransformer2DModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import is_torch_xla_available
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class DiTPipeline(DiffusionPipeline):
    """
    基于 Transformer 的图像生成管道（而非传统 UNet）

    核心组件:
    - transformer: DiT模型，用于去噪
    - vae: 编码/解码图像
    - scheduler: 控制去噪步骤
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer: DiTTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)

        # 构建 ImageNet 标签字典（标签名 -> 类别ID）
        self.labels = {}
        if id2label is not None:
            for key, value in id2label.items():
                for label in value.split(","):
                    self.labels[label.strip()] = int(key)
            self.labels = dict(sorted(self.labels.items()))

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        """将标签名转换为类别ID"""
        if not isinstance(label, list):
            label = list(label)

        for l in label:
            if l not in self.labels:
                raise ValueError(f"{l} 不存在。可用标签: {self.labels}")

        return [self.labels[l] for l in label]

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],  # 要生成的图像类别
        guidance_scale: float = 4.0,  # 引导强度，>1 时启用 CFG
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 50,  # 去噪步数
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        生成图像的主函数

        使用示例:
        >>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
        >>> class_ids = pipe.get_label_ids(["white shark", "umbrella"])
        >>> images = pipe(class_labels=class_ids, num_inference_steps=25)
        """
        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        # 1. 初始化随机噪声
        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        # 2. 准备条件（Classifier-Free Guidance）
        class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        # 3. 迭代去噪
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 转换时间步为张量
            timesteps = t
            if not torch.is_tensor(timesteps):
                is_mps = latent_model_input.device.type == "mps"
                is_npu = latent_model_input.device.type == "npu"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if (is_mps or is_npu) else torch.float64
                else:
                    dtype = torch.int32 if (is_mps or is_npu) else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)

            timesteps = timesteps.expand(latent_model_input.shape[0])

            # 预测噪声
            noise_pred = self.transformer(
                latent_model_input, timestep=timesteps, class_labels=class_labels_input
            ).sample

            # 应用 Classifier-Free Guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                noise_pred = torch.cat([eps, rest], dim=1)

            # 如果模型预测了 sigma，只取噪声部分
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # 去噪一步: x_t -> x_{t-1}
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

            if XLA_AVAILABLE:
                xm.mark_step()

        # 4. 解码为图像
        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        # 5. 后处理
        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
