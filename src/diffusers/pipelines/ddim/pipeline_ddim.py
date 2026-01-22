import torch

class DDIMPipeline:
    """
    DDIM 图像生成管道

    核心组件：
    - unet: 用于去噪的 UNet2D 模型
    - scheduler: DDIM 调度器
    """

    def __init__(self, unet, scheduler):
        self.unet = unet
        # 确保调度器是 DDIM
        from ...schedulers import DDIMScheduler
        self.scheduler = DDIMScheduler.from_config(scheduler.config)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator = None,
        eta: float = 0.0,          # eta=0 对应 DDIM，eta=1 对应 DDPM
        num_inference_steps: int = 50,
        output_type: str = "pil",
    ):
        """
        生成图像

        参数:
            eta: 控制随机性，0=确定性（DDIM），1=随机（DDPM）

        示例:
            pipe = DDIMPipeline.from_pretrained("fusing/ddim-lsun-bedroom")
            image = pipe(eta=0.0, num_inference_steps=50)
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
            image = self.scheduler.step(model_output, t, image, eta=eta, generator=generator).prev_sample

        # 3. 后处理
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            from ..pipeline_utils import numpy_to_pil
            image = numpy_to_pil(image)

        return image
