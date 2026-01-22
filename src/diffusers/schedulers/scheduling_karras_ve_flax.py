from dataclasses import dataclass
from typing import Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
from jax import random

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils_flax import FlaxSchedulerMixin

@flax.struct.dataclass
class KarrasVeSchedulerState:
    # setable values
    num_inference_steps: Optional[int] = None
    timesteps: Optional[jnp.ndarray] = None
    schedule: Optional[jnp.ndarray] = None  # sigma(t_i)

    @classmethod
    def create(cls):
        return cls()

@dataclass
class FlaxKarrasVeOutput(BaseOutput):
    class KarrasVeSchedulerState:
    # setable values
    num_inference_steps: Optional[int] = None
    timesteps: Optional[jnp.ndarray] = None
    schedule: Optional[jnp.ndarray] = None  # sigma(t_i)

    @classmethod
    def create(cls):
        return cls()

@dataclass
class FlaxKarrasVeOutput(BaseOutput):
    
    class KarrasVeSchedulerState:
    # setable values
    num_inference_steps: Optional[int] = None
    timesteps: Optional[jnp.ndarray] = None
    schedule: Optional[jnp.ndarray] = None  # sigma(t_i)

    @classmethod
    def create(cls):
        return cls()

@dataclass
class FlaxKarrasVeOutput(BaseOutput):
    class KarrasVeSchedulerState:
    # setable values
    num_inference_steps: Optional[int] = None
    timesteps: Optional[jnp.ndarray] = None
    schedule: Optional[jnp.ndarray] = None  # sigma(t_i)

    @classmethod
    def create(cls):
        return cls()

@dataclass
class FlaxKarrasVeOutput(BaseOutput):


    prev_sample: jnp.ndarray
    derivative: jnp.ndarray
    state: KarrasVeSchedulerState

class FlaxKarrasVeScheduler(FlaxSchedulerMixin, ConfigMixin):
    
    class FlaxKarrasVeScheduler(FlaxSchedulerMixin, ConfigMixin):


    @property
    def has_state(self):
        return True

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
        pass

    def create_state(self):
        return KarrasVeSchedulerState.create()

    def set_timesteps(
        self, state: KarrasVeSchedulerState, num_inference_steps: int, shape: Tuple = ()
    ) -> KarrasVeSchedulerState:

        timesteps = jnp.arange(0, num_inference_steps)[::-1].copy()
        schedule = [
            (
                self.config.sigma_max**2
                * (self.config.sigma_min**2 / self.config.sigma_max**2) ** (i / (num_inference_steps - 1))
            )
            for i in timesteps
        ]

        return state.replace(
            num_inference_steps=num_inference_steps,
            schedule=jnp.array(schedule, dtype=jnp.float32),
            timesteps=timesteps,
        )

    def add_noise_to_input(
        self,
        state: KarrasVeSchedulerState,
        sample: jnp.ndarray,
        sigma: float,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, float]:

        if self.config.s_min <= sigma <= self.config.s_max:
            gamma = min(self.config.s_churn / state.num_inference_steps, 2**0.5 - 1)
        else:
            gamma = 0

        # sample eps ~ N(0, S_noise^2 * I)
        key = random.split(key, num=1)
        eps = self.config.s_noise * random.normal(key=key, shape=sample.shape)
        sigma_hat = sigma + gamma * sigma
        sample_hat = sample + ((sigma_hat**2 - sigma**2) ** 0.5 * eps)

        return sample_hat, sigma_hat

    def step(
        self,
        state: KarrasVeSchedulerState,
        model_output: jnp.ndarray,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxKarrasVeOutput, Tuple]:


        pred_original_sample = sample_hat + sigma_hat * model_output
        derivative = (sample_hat - pred_original_sample) / sigma_hat
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative

        if not return_dict:
            return (sample_prev, derivative, state)

        return FlaxKarrasVeOutput(prev_sample=sample_prev, derivative=derivative, state=state)

    def step_correct(
        self,
        state: KarrasVeSchedulerState,
        model_output: jnp.ndarray,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: jnp.ndarray,
        sample_prev: jnp.ndarray,
        derivative: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxKarrasVeOutput, Tuple]:

        pred_original_sample = sample_prev + sigma_prev * model_output
        derivative_corr = (sample_prev - pred_original_sample) / sigma_prev
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)

        if not return_dict:
            return (sample_prev, derivative, state)

        return FlaxKarrasVeOutput(prev_sample=sample_prev, derivative=derivative, state=state)

    def add_noise(self, state: KarrasVeSchedulerState, original_samples, noise, timesteps):
        raise NotImplementedError()
