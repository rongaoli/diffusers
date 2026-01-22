from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..schedulers.scheduling_utils import SchedulerMixin
from ..utils import BaseOutput, logging
from ..utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->SCM
class SCMSchedulerOutput(BaseOutput):
    
    class SCMSchedulerOutput(BaseOutput):


    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None

class SCMScheduler(SchedulerMixin, ConfigMixin):
    
    class SCMScheduler(SchedulerMixin, ConfigMixin):


    # _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "trigflow",
        sigma_data: float = 0.5,
    ):

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

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

    def set_timesteps(
        self,
        num_inference_steps: int,
        timesteps: torch.Tensor = None,
        device: Union[str, torch.device] = None,
        max_timesteps: float = 1.57080,
        intermediate_timesteps: float = 1.3,
    ):

        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        if timesteps is not None and len(timesteps) != num_inference_steps + 1:
            raise ValueError("If providing custom timesteps, `timesteps` must be of length `num_inference_steps + 1`.")

        if timesteps is not None and max_timesteps is not None:
            raise ValueError("If providing custom timesteps, `max_timesteps` should not be provided.")

        if timesteps is None and max_timesteps is None:
            raise ValueError("Should provide either `timesteps` or `max_timesteps`.")

        if intermediate_timesteps is not None and num_inference_steps != 2:
            raise ValueError("Intermediate timesteps for SCM is not supported when num_inference_steps != 2.")

        self.num_inference_steps = num_inference_steps

        if timesteps is not None:
            if isinstance(timesteps, list):
                self.timesteps = torch.tensor(timesteps, device=device).float()
            elif isinstance(timesteps, torch.Tensor):
                self.timesteps = timesteps.to(device).float()
            else:
                raise ValueError(f"Unsupported timesteps type: {type(timesteps)}")
        elif intermediate_timesteps is not None:
            self.timesteps = torch.tensor([max_timesteps, intermediate_timesteps, 0], device=device).float()
        else:
            # max_timesteps=arctan(80/0.5)=1.56454...
            self.timesteps = torch.linspace(max_timesteps, 0, num_inference_steps + 1, device=device).float()

        self._step_index = None
        self._begin_index = None

    # Copied from diffusers.schedulers.scheduling_...
    def _init_step_index(self, timestep: Union[float, torch.Tensor]) -> None:

        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

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

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float,
        sample: torch.FloatTensor,
        generator: torch.Generator = None,
        return_dict: bool = True,
    ) -> Union[SCMSchedulerOutput, Tuple]:

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # 2. compute alphas, betas
        t = self.timesteps[self.step_index + 1]
        s = self.timesteps[self.step_index]

        # 4. Different Parameterization:
        parameterization = self.config.prediction_type

        if parameterization == "trigflow":
            pred_x0 = torch.cos(s) * sample - torch.sin(s) * model_output
        else:
            raise ValueError(f"Unsupported parameterization: {parameterization}")

        # 5. Sample z ~ N(0, I), For MultiStep Inference
        # Noise is not used for one-step sampling.
        if len(self.timesteps) > 1:
            noise = (
                randn_tensor(model_output.shape, device=model_output.device, generator=generator)
                * self.config.sigma_data
            )
            prev_sample = torch.cos(t) * pred_x0 + torch.sin(t) * noise
        else:
            prev_sample = pred_x0

        self._step_index += 1

        if not return_dict:
            return (prev_sample, pred_x0)

        return SCMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)

    def __len__(self):
        return self.config.num_train_timesteps
