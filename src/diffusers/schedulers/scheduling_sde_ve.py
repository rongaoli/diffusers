import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin, SchedulerOutput

@dataclass
class SdeVeOutput(BaseOutput):
    
    class SdeVeOutput(BaseOutput):


    prev_sample: torch.Tensor
    prev_sample_mean: torch.Tensor

class ScoreSdeVeScheduler(SchedulerMixin, ConfigMixin):
    
    class ScoreSdeVeScheduler(SchedulerMixin, ConfigMixin):


    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 2000,
        snr: float = 0.15,
        sigma_min: float = 0.01,
        sigma_max: float = 1348.0,
        sampling_eps: float = 1e-5,
        correct_steps: int = 1,
    ):
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        # setable values
        self.timesteps = None

        self.set_sigmas(num_train_timesteps, sigma_min, sigma_max, sampling_eps)

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:

        return sample

    def set_timesteps(
        self, num_inference_steps: int, sampling_eps: float = None, device: Union[str, torch.device] = None
    ):

        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps

        self.timesteps = torch.linspace(1, sampling_eps, num_inference_steps, device=device)

    def set_sigmas(
        self, num_inference_steps: int, sigma_min: float = None, sigma_max: float = None, sampling_eps: float = None
    ):

        sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        if self.timesteps is None:
            self.set_timesteps(num_inference_steps, sampling_eps)

        self.sigmas = sigma_min * (sigma_max / sigma_min) ** (self.timesteps / sampling_eps)
        self.discrete_sigmas = torch.exp(torch.linspace(math.log(sigma_min), math.log(sigma_max), num_inference_steps))
        self.sigmas = torch.tensor([sigma_min * (sigma_max / sigma_min) ** t for t in self.timesteps])

    def get_adjacent_sigma(self, timesteps, t):
        return torch.where(
            timesteps == 0,
            torch.zeros_like(t.to(timesteps.device)),
            self.discrete_sigmas[timesteps - 1].to(timesteps.device),
        )

    def step_pred(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SdeVeOutput, Tuple]:

        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        timestep = timestep * torch.ones(
            sample.shape[0], device=sample.device
        )  # torch.repeat_interleave(timestep, sample.shape[0])
        timesteps = (timestep * (len(self.timesteps) - 1)).long()

        # mps requires indices to be in the same device, so we use cpu as is the default with cuda
        timesteps = timesteps.to(self.discrete_sigmas.device)

        sigma = self.discrete_sigmas[timesteps].to(sample.device)
        adjacent_sigma = self.get_adjacent_sigma(timesteps, timestep).to(sample.device)
        drift = torch.zeros_like(sample)
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        # equation 6 in the paper: the model_output modeled by the network is grad_x log pt(x)
        # also equation 47 shows the analog from SDE models to ancestral sampling methods
        diffusion = diffusion.flatten()
        while len(diffusion.shape) < len(sample.shape):
            diffusion = diffusion.unsqueeze(-1)
        drift = drift - diffusion**2 * model_output

        #  equation 6: sample noise for the diffusion term of
        noise = randn_tensor(
            sample.shape, layout=sample.layout, generator=generator, device=sample.device, dtype=sample.dtype
        )
        prev_sample_mean = sample - drift  # subtract because `dt` is a small negative timestep
        # TODO is the variable diffusion the correct scaling term for the noise?
        prev_sample = prev_sample_mean + diffusion * noise  # add impact of diffusion field g

        if not return_dict:
            return (prev_sample, prev_sample_mean)

        return SdeVeOutput(prev_sample=prev_sample, prev_sample_mean=prev_sample_mean)

    def step_correct(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:

        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # For small batch sizes, the paper "sugges...
        # sample noise for correction
        noise = randn_tensor(sample.shape, layout=sample.layout, generator=generator).to(sample.device)

        # compute step size from the model_output, the noise, and the snr
        grad_norm = torch.norm(model_output.reshape(model_output.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (self.config.snr * noise_norm / grad_norm) ** 2 * 2
        step_size = step_size * torch.ones(sample.shape[0]).to(sample.device)
        # self.repeat_scalar(step_size, sample.shape[0])

        # compute corrected sample: model_output term and noise term
        step_size = step_size.flatten()
        while len(step_size.shape) < len(sample.shape):
            step_size = step_size.unsqueeze(-1)
        prev_sample_mean = sample + step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        timesteps = timesteps.to(original_samples.device)
        sigmas = self.discrete_sigmas.to(original_samples.device)[timesteps]
        noise = (
            noise * sigmas[:, None, None, None]
            if noise is not None
            else torch.randn_like(original_samples) * sigmas[:, None, None, None]
        )
        noisy_samples = noise + original_samples
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
