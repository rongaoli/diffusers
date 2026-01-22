from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class CMStochasticIterativeSchedulerOutput(BaseOutput):
    
    class CMStochasticIterativeSchedulerOutput(BaseOutput):


    prev_sample: torch.Tensor

class CMStochasticIterativeScheduler(SchedulerMixin, ConfigMixin):
    
    class CMStochasticIterativeScheduler(SchedulerMixin, ConfigMixin):


    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 40,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        s_noise: float = 1.0,
        rho: float = 7.0,
        clip_denoised: bool = True,
    ) -> None:
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        ramp = np.linspace(0, 1, num_train_timesteps)
        sigmas = self._convert_to_karras(ramp)
        timesteps = self.sigma_to_t(sigmas)

        # setable values
        self.num_inference_steps = None
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps)
        self.custom_timesteps = False
        self.is_scale_input_called = False
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def step_index(self) -> Optional[int]:
        The index counter for current timestep. It will increase 1 after each scheduler step.
