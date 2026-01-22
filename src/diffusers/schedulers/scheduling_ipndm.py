import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin, SchedulerOutput

class IPNDMScheduler(SchedulerMixin, ConfigMixin):
    
    class IPNDMScheduler(SchedulerMixin, ConfigMixin):


    order = 1

    @register_to_config
    def __init__(
        self, num_train_timesteps: int = 1000, trained_betas: Optional[Union[np.ndarray, List[float]]] = None
    ):
        # set `betas`, `alphas`, `timesteps`
        self.set_timesteps(num_train_timesteps)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm pl...
        # mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

        # running values
        self.ets = []
        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self):

        return self._step_index

    @property
    def begin_index(self):

        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_...
    def set_begin_index(self, begin_index: int = 0):

        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):

        self.num_inference_steps = num_inference_steps
        steps = torch.linspace(1, 0, num_inference_steps + 1)[:-1]
        steps = torch.cat([steps, torch.tensor([0.0])])

        if self.config.trained_betas is not None:
            self.betas = torch.tensor(self.config.trained_betas, dtype=torch.float32)
        else:
            self.betas = torch.sin(steps * math.pi / 2) ** 2

        self.alphas = (1.0 - self.betas**2) ** 0.5

        timesteps = (torch.atan2(self.betas, self.alphas) / math.pi * 2)[:-1]
        self.timesteps = timesteps.to(device)

        self.ets = []
        self._step_index = None
        self._begin_index = None

    # Copied from diffusers.schedulers.scheduling_...
    def index_for_timestep(
        self, timestep: Union[float, torch.Tensor], schedule_timesteps: Optional[torch.Tensor] = None
    ) -> int:

        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    # Copied from diffusers.schedulers.scheduling_...
    def _init_step_index(self, timestep: Union[float, torch.Tensor]) -> None:

        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        if self.step_index is None:
            self._init_step_index(timestep)

        timestep_index = self.step_index
        prev_timestep_index = self.step_index + 1

        ets = sample * self.betas[timestep_index] + model_output * self.alphas[timestep_index]
        self.ets.append(ets)

        if len(self.ets) == 1:
            ets = self.ets[-1]
        elif len(self.ets) == 2:
            ets = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            ets = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        else:
            ets = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])

        prev_sample = self._get_prev_sample(sample, timestep_index, prev_timestep_index, ets)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        return sample

    def _get_prev_sample(self, sample, timestep_index, prev_timestep_index, ets):
        alpha = self.alphas[timestep_index]
        sigma = self.betas[timestep_index]

        next_alpha = self.alphas[prev_timestep_index]
        next_sigma = self.betas[prev_timestep_index]

        pred = (sample - sigma * ets) / max(alpha, 1e-8)
        prev_sample = next_alpha * pred + ets * next_sigma

        return prev_sample

    def __len__(self):
        return self.config.num_train_timesteps
