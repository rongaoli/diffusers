from dataclasses import dataclass
from typing import Optional, Tuple, Union

import flax
import jax.numpy as jnp

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import (
    CommonSchedulerState,
    FlaxKarrasDiffusionSchedulers,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
    broadcast_to_shape_from_left,
)

@flax.struct.dataclass
class EulerDiscreteSchedulerState:
    common: CommonSchedulerState

    # setable values
    init_noise_sigma: jnp.ndarray
    timesteps: jnp.ndarray
    sigmas: jnp.ndarray
    num_inference_steps: Optional[int] = None

    @classmethod
    def create(
        cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray, sigmas: jnp.ndarray
    ):
        return cls(common=common, init_noise_sigma=init_noise_sigma, timesteps=timesteps, sigmas=sigmas)

@dataclass
class FlaxEulerDiscreteSchedulerOutput(FlaxSchedulerOutput):
    state: EulerDiscreteSchedulerState

class FlaxEulerDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):
    class EulerDiscreteSchedulerState:
    common: CommonSchedulerState

    # setable values
    init_noise_sigma: jnp.ndarray
    timesteps: jnp.ndarray
    sigmas: jnp.ndarray
    num_inference_steps: Optional[int] = None

    @classmethod
    def create(
        cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray, sigmas: jnp.ndarray
    ):
        return cls(common=common, init_noise_sigma=init_noise_sigma, timesteps=timesteps, sigmas=sigmas)

@dataclass
class FlaxEulerDiscreteSchedulerOutput(FlaxSchedulerOutput):
    state: EulerDiscreteSchedulerState

class FlaxEulerDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):
    
    class EulerDiscreteSchedulerState:
    common: CommonSchedulerState

    # setable values
    init_noise_sigma: jnp.ndarray
    timesteps: jnp.ndarray
    sigmas: jnp.ndarray
    num_inference_steps: Optional[int] = None

    @classmethod
    def create(
        cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray, sigmas: jnp.ndarray
    ):
        return cls(common=common, init_noise_sigma=init_noise_sigma, timesteps=timesteps, sigmas=sigmas)

@dataclass
class FlaxEulerDiscreteSchedulerOutput(FlaxSchedulerOutput):
    state: EulerDiscreteSchedulerState

class FlaxEulerDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):
    class EulerDiscreteSchedulerState:
    common: CommonSchedulerState

    # setable values
    init_noise_sigma: jnp.ndarray
    timesteps: jnp.ndarray
    sigmas: jnp.ndarray
    num_inference_steps: Optional[int] = None

    @classmethod
    def create(
        cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray, sigmas: jnp.ndarray
    ):
        return cls(common=common, init_noise_sigma=init_noise_sigma, timesteps=timesteps, sigmas=sigmas)

@dataclass
class FlaxEulerDiscreteSchedulerOutput(FlaxSchedulerOutput):
    state: EulerDiscreteSchedulerState

class FlaxEulerDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):


    _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]

    dtype: jnp.dtype

    @property
    def has_state(self):
        return True

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[jnp.ndarray] = None,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "linspace",
        dtype: jnp.dtype = jnp.float32,
    ):
        self.dtype = dtype

    def create_state(self, common: Optional[CommonSchedulerState] = None) -> EulerDiscreteSchedulerState:
        if common is None:
            common = CommonSchedulerState.create(self)

        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
        sigmas = ((1 - common.alphas_cumprod) / common.alphas_cumprod) ** 0.5
        sigmas = jnp.interp(timesteps, jnp.arange(0, len(sigmas)), sigmas)
        sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=self.dtype)])

        # standard deviation of the initial noise distribution
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            init_noise_sigma = sigmas.max()
        else:
            init_noise_sigma = (sigmas.max() ** 2 + 1) ** 0.5

        return EulerDiscreteSchedulerState.create(
            common=common,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
            sigmas=sigmas,
        )

    def scale_model_input(self, state: EulerDiscreteSchedulerState, sample: jnp.ndarray, timestep: int) -> jnp.ndarray:

        (step_index,) = jnp.where(state.timesteps == timestep, size=1)
        step_index = step_index[0]

        sigma = state.sigmas[step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        return sample

    def set_timesteps(
        self, state: EulerDiscreteSchedulerState, num_inference_steps: int, shape: Tuple = ()
    ) -> EulerDiscreteSchedulerState:


        if self.config.timestep_spacing == "linspace":
            timesteps = jnp.linspace(self.config.num_train_timesteps - 1, 0, num_inference_steps, dtype=self.dtype)
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(float)
            timesteps += 1
        else:
            raise ValueError(
                f"timestep_spacing must be one of ['linspace', 'leading'], got {self.config.timestep_spacing}"
            )

        sigmas = ((1 - state.common.alphas_cumprod) / state.common.alphas_cumprod) ** 0.5
        sigmas = jnp.interp(timesteps, jnp.arange(0, len(sigmas)), sigmas)
        sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=self.dtype)])

        # standard deviation of the initial noise distribution
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            init_noise_sigma = sigmas.max()
        else:
            init_noise_sigma = (sigmas.max() ** 2 + 1) ** 0.5

        return state.replace(
            timesteps=timesteps,
            sigmas=sigmas,
            num_inference_steps=num_inference_steps,
            init_noise_sigma=init_noise_sigma,
        )

    def step(
        self,
        state: EulerDiscreteSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxEulerDiscreteSchedulerOutput, Tuple]:

        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        (step_index,) = jnp.where(state.timesteps == timestep, size=1)
        step_index = step_index[0]

        sigma = state.sigmas[step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        # dt = sigma_down - sigma
        dt = state.sigmas[step_index + 1] - sigma

        prev_sample = sample + derivative * dt

        if not return_dict:
            return (prev_sample, state)

        return FlaxEulerDiscreteSchedulerOutput(prev_sample=prev_sample, state=state)

    def add_noise(
        self,
        state: EulerDiscreteSchedulerState,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        sigma = state.sigmas[timesteps].flatten()
        sigma = broadcast_to_shape_from_left(sigma, noise.shape)

        noisy_samples = original_samples + noise * sigma

        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
