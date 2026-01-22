import math
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_dpmsolver_sde import BrownianTreeNoiseSampler
from .scheduling_utils import SchedulerMixin, SchedulerOutput

class CosineDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    
    class CosineDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):


    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.3,
        sigma_max: float = 500,
        sigma_data: float = 1.0,
        sigma_schedule: Literal["exponential", "karras"] = "exponential",
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: Literal["epsilon", "sample", "v_prediction"] = "v_prediction",
        rho: float = 7.0,
        solver_type: Literal["midpoint", "heun"] = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        final_sigmas_type: Literal["zero", "sigma_min"] = "zero",
    ) -> None:
        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        ramp = torch.linspace(0, 1, num_train_timesteps)
        if sigma_schedule == "karras":
            sigmas = self._compute_karras_sigmas(ramp)
        elif sigma_schedule == "exponential":
            sigmas = self._compute_exponential_sigmas(ramp)

        self.timesteps = self.precondition_noise(sigmas)

        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        # setable values
        self.num_inference_steps = None
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def init_noise_sigma(self) -> float:
        The standard deviation of the initial noise distribution.
