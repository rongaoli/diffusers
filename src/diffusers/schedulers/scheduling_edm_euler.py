import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->EulerDiscrete
class EDMEulerSchedulerOutput(BaseOutput):
    
    class EDMEulerSchedulerOutput(BaseOutput):


    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None

class EDMEulerScheduler(SchedulerMixin, ConfigMixin):
    
    class EDMEulerScheduler(SchedulerMixin, ConfigMixin):


    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        sigma_schedule: Literal["karras", "exponential"] = "karras",
        num_train_timesteps: int = 1000,
        prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
        rho: float = 7.0,
        final_sigmas_type: Literal["zero", "sigma_min"] = "zero",
    ) -> None:
        if sigma_schedule not in ["karras", "exponential"]:
            raise ValueError(f"Wrong value for provided for `{sigma_schedule=}`.`")

        # setable values
        self.num_inference_steps = None

        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        sigmas = torch.arange(num_train_timesteps + 1, dtype=sigmas_dtype) / num_train_timesteps
        if sigma_schedule == "karras":
            sigmas = self._compute_karras_sigmas(sigmas)
        elif sigma_schedule == "exponential":
            sigmas = self._compute_exponential_sigmas(sigmas)
        sigmas = sigmas.to(torch.float32)

        self.timesteps = self.precondition_noise(sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = sigmas[-1]
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        self.sigmas = torch.cat([sigmas, torch.full((1,), fill_value=sigma_last, device=sigmas.device)])

        self.is_scale_input_called = False

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def init_noise_sigma(self) -> float:
        Return the standard deviation of the initial noise distribution.
