import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    max_beta: float = 0.999,
    alpha_transform_type: Literal["cosine", "exp", "laplace"] = "cosine",
) -> torch.Tensor:

    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "laplace":

        def alpha_bar_fn(t):
            lmb = -0.5 * math.copysign(1, 0.5 - t) * math.log(1 - 2 * math.fabs(0.5 - t) + 1e-6)
            snr = math.exp(lmb)
            return math.sqrt(snr / (1 + snr))

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

@dataclass
class ConsistencyDecoderSchedulerOutput(BaseOutput):
    
    class ConsistencyDecoderSchedulerOutput(BaseOutput):


    prev_sample: torch.Tensor

class ConsistencyDecoderScheduler(SchedulerMixin, ConfigMixin):
    
    class ConsistencyDecoderScheduler(SchedulerMixin, ConfigMixin):


    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1024,
        sigma_data: float = 0.5,
    ) -> None:
        betas = betas_for_alpha_bar(num_train_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        sigmas = torch.sqrt(1.0 / alphas_cumprod - 1)

        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)

        self.c_skip = sqrt_recip_alphas_cumprod * sigma_data**2 / (sigmas**2 + sigma_data**2)
        self.c_out = sigmas * sigma_data / (sigmas**2 + sigma_data**2) ** 0.5
        self.c_in = sqrt_recip_alphas_cumprod / (sigmas**2 + sigma_data**2) ** 0.5

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:

        if num_inference_steps != 2:
            raise ValueError("Currently more than 2 inference steps are not supported.")

        self.timesteps = torch.tensor([1008, 512], dtype=torch.long, device=device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.c_skip = self.c_skip.to(device)
        self.c_out = self.c_out.to(device)
        self.c_in = self.c_in.to(device)

    @property
    def init_noise_sigma(self) -> torch.Tensor:
        Return the standard deviation of the initial noise distribution.
