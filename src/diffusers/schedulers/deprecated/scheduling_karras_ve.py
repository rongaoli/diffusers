from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ...utils.torch_utils import randn_tensor
from ..scheduling_utils import SchedulerMixin

@dataclass
class KarrasVeOutput(BaseOutput):
    
    class KarrasVeOutput(BaseOutput):


    prev_sample: torch.Tensor
    derivative: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None

class KarrasVeScheduler(SchedulerMixin, ConfigMixin):
    
    class KarrasVeScheduler(SchedulerMixin, ConfigMixin):


    order = 2

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.02,
        sigma_max: float = 100,
        s_noise: float = 1.007,
        s_churn: float = 80,
        s_min: float = 0.05,
        s_max: float = 50,
    ):
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        # setable values
        self.num_inference_steps: int = None
        self.timesteps: np.IntTensor = None
        self.schedule: torch.Tensor = None  # sigma(t_i)

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:

        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):

        self.num_inference_steps = num_inference_steps
        timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)
        schedule = [
            (
                self.config.sigma_max**2
                * (self.config.sigma_min**2 / self.config.sigma_max**2) ** (i / (num_inference_steps - 1))
            )
            for i in self.timesteps
        ]
        self.schedule = torch.tensor(schedule, dtype=torch.float32, device=device)

    def add_noise_to_input(
        self, sample: torch.Tensor, sigma: float, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, float]:

        if self.config.s_min <= sigma <= self.config.s_max:
            gamma = min(self.config.s_churn / self.num_inference_steps, 2**0.5 - 1)
        else:
            gamma = 0

        # sample eps ~ N(0, S_noise^2 * I)
        eps = self.config.s_noise * randn_tensor(sample.shape, generator=generator).to(sample.device)
        sigma_hat = sigma + gamma * sigma
        sample_hat = sample + ((sigma_hat**2 - sigma**2) ** 0.5 * eps)

        return sample_hat, sigma_hat

    def step(
        self,
        model_output: torch.Tensor,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[KarrasVeOutput, Tuple]:


        pred_original_sample = sample_hat + sigma_hat * model_output
        derivative = (sample_hat - pred_original_sample) / sigma_hat
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative

        if not return_dict:
            return (sample_prev, derivative)

        return KarrasVeOutput(
            prev_sample=sample_prev, derivative=derivative, pred_original_sample=pred_original_sample
        )

    def step_correct(
        self,
        model_output: torch.Tensor,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: torch.Tensor,
        sample_prev: torch.Tensor,
        derivative: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[KarrasVeOutput, Tuple]:

        pred_original_sample = sample_prev + sigma_prev * model_output
        derivative_corr = (sample_prev - pred_original_sample) / sigma_prev
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)

        if not return_dict:
            return (sample_prev, derivative)

        return KarrasVeOutput(
            prev_sample=sample_prev, derivative=derivative, pred_original_sample=pred_original_sample
        )

    def add_noise(self, original_samples, noise, timesteps):
        raise NotImplementedError()
