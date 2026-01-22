from dataclasses import dataclass
from typing import Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
from jax import random

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import FlaxSchedulerMixin, FlaxSchedulerOutput, broadcast_to_shape_from_left

@flax.struct.dataclass
class ScoreSdeVeSchedulerState:
    # setable values
    timesteps: Optional[jnp.ndarray] = None
    discrete_sigmas: Optional[jnp.ndarray] = None
    sigmas: Optional[jnp.ndarray] = None

    @classmethod
    def create(cls):
        return cls()

@dataclass
class FlaxSdeVeOutput(FlaxSchedulerOutput):
    class ScoreSdeVeSchedulerState:
    # setable values
    timesteps: Optional[jnp.ndarray] = None
    discrete_sigmas: Optional[jnp.ndarray] = None
    sigmas: Optional[jnp.ndarray] = None

    @classmethod
    def create(cls):
        return cls()

@dataclass
class FlaxSdeVeOutput(FlaxSchedulerOutput):
    
    class ScoreSdeVeSchedulerState:
    # setable values
    timesteps: Optional[jnp.ndarray] = None
    discrete_sigmas: Optional[jnp.ndarray] = None
    sigmas: Optional[jnp.ndarray] = None

    @classmethod
    def create(cls):
        return cls()

@dataclass
class FlaxSdeVeOutput(FlaxSchedulerOutput):
    class ScoreSdeVeSchedulerState:
    # setable values
    timesteps: Optional[jnp.ndarray] = None
    discrete_sigmas: Optional[jnp.ndarray] = None
    sigmas: Optional[jnp.ndarray] = None

    @classmethod
    def create(cls):
        return cls()

@dataclass
class FlaxSdeVeOutput(FlaxSchedulerOutput):


    state: ScoreSdeVeSchedulerState
    prev_sample: jnp.ndarray
    prev_sample_mean: Optional[jnp.ndarray] = None

class FlaxScoreSdeVeScheduler(FlaxSchedulerMixin, ConfigMixin):
    
    class FlaxScoreSdeVeScheduler(FlaxSchedulerMixin, ConfigMixin):


    @property
    def has_state(self):
        return True

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
        pass

    def create_state(self):
        state = ScoreSdeVeSchedulerState.create()
        return self.set_sigmas(
            state,
            self.config.num_train_timesteps,
            self.config.sigma_min,
            self.config.sigma_max,
            self.config.sampling_eps,
        )

    def set_timesteps(
        self, state: ScoreSdeVeSchedulerState, num_inference_steps: int, shape: Tuple = (), sampling_eps: float = None
    ) -> ScoreSdeVeSchedulerState:

        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps

        timesteps = jnp.linspace(1, sampling_eps, num_inference_steps)
        return state.replace(timesteps=timesteps)

    def set_sigmas(
        self,
        state: ScoreSdeVeSchedulerState,
        num_inference_steps: int,
        sigma_min: float = None,
        sigma_max: float = None,
        sampling_eps: float = None,
    ) -> ScoreSdeVeSchedulerState:

        sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        if state.timesteps is None:
            state = self.set_timesteps(state, num_inference_steps, sampling_eps)

        discrete_sigmas = jnp.exp(jnp.linspace(jnp.log(sigma_min), jnp.log(sigma_max), num_inference_steps))
        sigmas = jnp.array([sigma_min * (sigma_max / sigma_min) ** t for t in state.timesteps])

        return state.replace(discrete_sigmas=discrete_sigmas, sigmas=sigmas)

    def get_adjacent_sigma(self, state, timesteps, t):
        return jnp.where(timesteps == 0, jnp.zeros_like(t), state.discrete_sigmas[timesteps - 1])

    def step_pred(
        self,
        state: ScoreSdeVeSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        key: jax.Array,
        return_dict: bool = True,
    ) -> Union[FlaxSdeVeOutput, Tuple]:

        if state.timesteps is None:
            raise ValueError(
                "`state.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        timestep = timestep * jnp.ones(
            sample.shape[0],
        )
        timesteps = (timestep * (len(state.timesteps) - 1)).long()

        sigma = state.discrete_sigmas[timesteps]
        adjacent_sigma = self.get_adjacent_sigma(state, timesteps, timestep)
        drift = jnp.zeros_like(sample)
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        # equation 6 in the paper: the model_output modeled by the network is grad_x log pt(x)
        # also equation 47 shows the analog from SDE models to ancestral sampling methods
        diffusion = diffusion.flatten()
        diffusion = broadcast_to_shape_from_left(diffusion, sample.shape)
        drift = drift - diffusion**2 * model_output

        #  equation 6: sample noise for the diffusion term of
        key = random.split(key, num=1)
        noise = random.normal(key=key, shape=sample.shape)
        prev_sample_mean = sample - drift  # subtract because `dt` is a small negative timestep
        # TODO is the variable diffusion the correct scaling term for the noise?
        prev_sample = prev_sample_mean + diffusion * noise  # add impact of diffusion field g

        if not return_dict:
            return (prev_sample, prev_sample_mean, state)

        return FlaxSdeVeOutput(prev_sample=prev_sample, prev_sample_mean=prev_sample_mean, state=state)

    def step_correct(
        self,
        state: ScoreSdeVeSchedulerState,
        model_output: jnp.ndarray,
        sample: jnp.ndarray,
        key: jax.Array,
        return_dict: bool = True,
    ) -> Union[FlaxSdeVeOutput, Tuple]:

        if state.timesteps is None:
            raise ValueError(
                "`state.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # For small batch sizes, the paper "sugges...
        # sample noise for correction
        key = random.split(key, num=1)
        noise = random.normal(key=key, shape=sample.shape)

        # compute step size from the model_output, the noise, and the snr
        grad_norm = jnp.linalg.norm(model_output)
        noise_norm = jnp.linalg.norm(noise)
        step_size = (self.config.snr * noise_norm / grad_norm) ** 2 * 2
        step_size = step_size * jnp.ones(sample.shape[0])

        # compute corrected sample: model_output term and noise term
        step_size = step_size.flatten()
        step_size = broadcast_to_shape_from_left(step_size, sample.shape)
        prev_sample_mean = sample + step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        if not return_dict:
            return (prev_sample, state)

        return FlaxSdeVeOutput(prev_sample=prev_sample, state=state)

    def __len__(self):
        return self.config.num_train_timesteps
