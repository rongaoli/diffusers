import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin

@dataclass
class DDPMWuerstchenSchedulerOutput(BaseOutput):
    
    class DDPMWuerstchenSchedulerOutput(BaseOutput):


    prev_sample: torch.Tensor

def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):

    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

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

class DDPMWuerstchenScheduler(SchedulerMixin, ConfigMixin):
    
    class DDPMWuerstchenScheduler(SchedulerMixin, ConfigMixin):


    @register_to_config
    def __init__(
        self,
        scaler: float = 1.0,
        s: float = 0.008,
    ):
        self.scaler = scaler
        self.s = torch.tensor([s])
        self._init_alpha_cumprod = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

    def _alpha_cumprod(self, t, device):
        if self.scaler > 1:
            t = 1 - (1 - t) ** self.scaler
        elif self.scaler < 1:
            t = t**self.scaler
        alpha_cumprod = torch.cos(
            (t + self.s.to(device)) / (1 + self.s.to(device)) * torch.pi * 0.5
        ) ** 2 / self._init_alpha_cumprod.to(device)
        return alpha_cumprod.clamp(0.0001, 0.9999)

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:

        return sample

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        timesteps: Optional[List[int]] = None,
        device: Union[str, torch.device] = None,
    ):

        if timesteps is None:
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.Tensor(timesteps).to(device)
        self.timesteps = timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMWuerstchenSchedulerOutput, Tuple]:

        dtype = model_output.dtype
        device = model_output.device
        t = timestep

        prev_t = self.previous_timestep(t)

        alpha_cumprod = self._alpha_cumprod(t, device).view(t.size(0), *[1 for _ in sample.shape[1:]])
        alpha_cumprod_prev = self._alpha_cumprod(prev_t, device).view(prev_t.size(0), *[1 for _ in sample.shape[1:]])
        alpha = alpha_cumprod / alpha_cumprod_prev

        mu = (1.0 / alpha).sqrt() * (sample - (1 - alpha) * model_output / (1 - alpha_cumprod).sqrt())

        std_noise = randn_tensor(mu.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
        std = ((1 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)).sqrt() * std_noise
        pred = mu + std * (prev_t != 0).float().view(prev_t.size(0), *[1 for _ in sample.shape[1:]])

        if not return_dict:
            return (pred.to(dtype),)

        return DDPMWuerstchenSchedulerOutput(prev_sample=pred.to(dtype))

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        device = original_samples.device
        dtype = original_samples.dtype
        alpha_cumprod = self._alpha_cumprod(timesteps, device=device).view(
            timesteps.size(0), *[1 for _ in original_samples.shape[1:]]
        )
        noisy_samples = alpha_cumprod.sqrt() * original_samples + (1 - alpha_cumprod).sqrt() * noise
        return noisy_samples.to(dtype=dtype)

    def __len__(self):
        return self.config.num_train_timesteps

    def previous_timestep(self, timestep):
        index = (self.timesteps - timestep[0]).abs().argmin().item()
        prev_t = self.timesteps[index + 1][None].expand(timestep.shape[0])
        return prev_t
