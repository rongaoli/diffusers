import importlib
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import flax
import jax.numpy as jnp
from huggingface_hub.utils import validate_hf_hub_args

from ..utils import BaseOutput, PushToHubMixin, logging

logger = logging.get_logger(__name__)

SCHEDULER_CONFIG_NAME = "scheduler_config.json"

# NOTE: We make this type an enum because it simplifies usage in docs and prevents
# circular imports when used for `_compatibles` within the schedulers module.
# When it's used as a type in pipelines, it really is a Union because the actual
# scheduler instance is passed in.
class FlaxKarrasDiffusionSchedulers(Enum):
    FlaxDDIMScheduler = 1
    FlaxDDPMScheduler = 2
    FlaxPNDMScheduler = 3
    FlaxLMSDiscreteScheduler = 4
    FlaxDPMSolverMultistepScheduler = 5
    FlaxEulerDiscreteScheduler = 6

@dataclass
class FlaxSchedulerOutput(BaseOutput):
    class FlaxKarrasDiffusionSchedulers(Enum):
    FlaxDDIMScheduler = 1
    FlaxDDPMScheduler = 2
    FlaxPNDMScheduler = 3
    FlaxLMSDiscreteScheduler = 4
    FlaxDPMSolverMultistepScheduler = 5
    FlaxEulerDiscreteScheduler = 6

@dataclass
class FlaxSchedulerOutput(BaseOutput):
    
    class FlaxKarrasDiffusionSchedulers(Enum):
    FlaxDDIMScheduler = 1
    FlaxDDPMScheduler = 2
    FlaxPNDMScheduler = 3
    FlaxLMSDiscreteScheduler = 4
    FlaxDPMSolverMultistepScheduler = 5
    FlaxEulerDiscreteScheduler = 6

@dataclass
class FlaxSchedulerOutput(BaseOutput):
    class FlaxKarrasDiffusionSchedulers(Enum):
    FlaxDDIMScheduler = 1
    FlaxDDPMScheduler = 2
    FlaxPNDMScheduler = 3
    FlaxLMSDiscreteScheduler = 4
    FlaxDPMSolverMultistepScheduler = 5
    FlaxEulerDiscreteScheduler = 6

@dataclass
class FlaxSchedulerOutput(BaseOutput):


    prev_sample: jnp.ndarray

class FlaxSchedulerMixin(PushToHubMixin):
    
    class FlaxSchedulerMixin(PushToHubMixin):


    config_name = SCHEDULER_CONFIG_NAME
    ignore_for_config = ["dtype"]
    _compatibles = []
    has_compatibles = True

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        subfolder: Optional[str] = None,
        return_unused_kwargs=False,
        **kwargs,
    ):

        logger.warning(
            "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We "
            "recommend migrating to PyTorch classes or pinning your version of Diffusers."
        )
        config, kwargs = cls.load_config(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            **kwargs,
        )
        scheduler, unused_kwargs = cls.from_config(config, return_unused_kwargs=True, **kwargs)

        if hasattr(scheduler, "create_state") and getattr(scheduler, "has_state", False):
            state = scheduler.create_state()

        if return_unused_kwargs:
            return scheduler, state, unused_kwargs

        return scheduler, state

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        class from a pre-defined JSON-file.

                    - A string, the *model id* of a model repo on huggingface.co. Valid model ids should have an
                      organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~SchedulerMixin.save_pretrained`],
                      e.g., `./my_model_directory/`.
            subfolder (`str`, *optional*):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

        > [!TIP] > It is required to be logged in (`hf auth login`) when you want to use private or [gated >
        models](https://huggingface.co/docs/hub/models-gated#gated-models).

        > [!TIP] > Activate the special
        ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to > use this method in a
        firewalled environment.

        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)

    @property
    def compatibles(self):
        Returns all schedulers that are compatible with this scheduler
class CommonSchedulerState:
    alphas: jnp.ndarray
    betas: jnp.ndarray
    alphas_cumprod: jnp.ndarray

    @classmethod
    def create(cls, scheduler):
        config = scheduler.config

        if config.trained_betas is not None:
            betas = jnp.asarray(config.trained_betas, dtype=scheduler.dtype)
        elif config.beta_schedule == "linear":
            betas = jnp.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=scheduler.dtype)
        elif config.beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (
                jnp.linspace(
                    config.beta_start**0.5, config.beta_end**0.5, config.num_train_timesteps, dtype=scheduler.dtype
                )
                ** 2
            )
        elif config.beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(config.num_train_timesteps, dtype=scheduler.dtype)
        else:
            raise NotImplementedError(
                f"beta_schedule {config.beta_schedule} is not implemented for scheduler {scheduler.__class__.__name__}"
            )

        alphas = 1.0 - betas

        alphas_cumprod = jnp.cumprod(alphas, axis=0)

        return cls(
            alphas=alphas,
            betas=betas,
            alphas_cumprod=alphas_cumprod,
        )

def get_sqrt_alpha_prod(
    state: CommonSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray
):
    alphas_cumprod = state.alphas_cumprod

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    sqrt_alpha_prod = broadcast_to_shape_from_left(sqrt_alpha_prod, original_samples.shape)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    sqrt_one_minus_alpha_prod = broadcast_to_shape_from_left(sqrt_one_minus_alpha_prod, original_samples.shape)

    return sqrt_alpha_prod, sqrt_one_minus_alpha_prod

def add_noise_common(
    state: CommonSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray
):
    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(state, original_samples, noise, timesteps)
    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples

def get_velocity_common(state: CommonSchedulerState, sample: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray):
    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(state, sample, noise, timesteps)
    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    return velocity
