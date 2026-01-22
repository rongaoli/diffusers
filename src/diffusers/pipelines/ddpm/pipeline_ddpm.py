import torch

class DDPMPipeline:
    """
    DDPM 图像生成管道

    核心组件：
    - unet: 用于去噪的 UNet2D 模型
    - scheduler: DDPM 调度器
    """

    def __init__(self, unet, scheduler):
        self.unet = unet
        self.scheduler = scheduler

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator = None,
        num_inference_steps: int = 1000,
        output_type: str = "pil",
    ):
        """
        生成图像

        示例:
            pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")
            image = pipe().images[0]
        """
        # 1. 初始化随机噪声
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        image = torch.randn(image_shape, generator=generator, device=self.unet.device, dtype=self.unet.dtype)

        # 2. 迭代去噪
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            # 预测噪声
            model_output = self.unet(image, t).sample
            # 去噪一步：x_t -> x_{t-1}
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # 3. 后处理
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            from ..pipeline_utils import numpy_to_pil
            image = numpy_to_pil(image)

        return image
