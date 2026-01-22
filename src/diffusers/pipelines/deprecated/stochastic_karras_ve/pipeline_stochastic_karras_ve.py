from typing import List, Optional, Tuple, Union

import torch

from ....models import UNet2DModel
from ....schedulers import KarrasVeScheduler
from ....utils.torch_utils import randn_tensor
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput

class KarrasVePipeline(DiffusionPipeline):


    # add type hints for linting
    unet: UNet2DModel
    scheduler: KarrasVeScheduler

    def __init__(self, unet: UNet2DModel, scheduler: KarrasVeScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        
        """r"""

        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet

        # sample x_0 ~ N(0, sigma_0^2 * I)
        sample = randn_tensor(shape, generator=generator, device=self.device) * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # here sigma_t == t_i from the paper
            sigma = self.scheduler.schedule[t]
            sigma_prev = self.scheduler.schedule[t - 1] if t > 0 else 0

            # 1. Select temporarily increased noise level sigma_hat
            # 2. Add new noise to move from sample_i to sample_hat
            sample_hat, sigma_hat = self.scheduler.add_noise_to_input(sample, sigma, generator=generator)

            # 3. Predict the noise residual given the noise magnitude `sigma_hat`
            # The model inputs and output are adjusted by following eq. (213) in [1].
            model_output = (sigma_hat / 2) * model((sample_hat + 1) / 2, sigma_hat / 2).sample

            # 4. Evaluate dx/dt at sigma_hat
            # 5. Take Euler step from sigma to sigma_prev
            step_output = self.scheduler.step(model_output, sigma_hat, sigma_prev, sample_hat)

            if sigma_prev != 0:
                # 6. Apply 2nd order correction
                # The model inputs and output are adjusted by following eq. (213) in [1].
                model_output = (sigma_prev / 2) * model((step_output.prev_sample + 1) / 2, sigma_prev / 2).sample
                step_output = self.scheduler.step_correct(
                    model_output,
                    sigma_hat,
                    sigma_prev,
                    sample_hat,
                    step_output.prev_sample,
                    step_output["derivative"],
                )
            sample = step_output.prev_sample

        sample = (sample / 2 + 0.5).clamp(0, 1)
        image = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
