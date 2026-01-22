import importlib
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
from huggingface_hub.utils import validate_hf_hub_args
from typing_extensions import Self

from ..utils import BaseOutput, PushToHubMixin

SCHEDULER_CONFIG_NAME = "scheduler_config.json"

# NOTE: We make this type an enum because it simplifies usage in docs and prevents
# circular imports when used for `_compatibles` within the schedulers module.
# When it's used as a type in pipelines, it really is a Union because the actual
# scheduler instance is passed in.
class KarrasDiffusionSchedulers(Enum):
    DDIMScheduler = 1
    DDPMScheduler = 2
    PNDMScheduler = 3
    LMSDiscreteScheduler = 4
    EulerDiscreteScheduler = 5
    HeunDiscreteScheduler = 6
    EulerAncestralDiscreteScheduler = 7
    DPMSolverMultistepScheduler = 8
    DPMSolverSinglestepScheduler = 9
    KDPM2DiscreteScheduler = 10
    KDPM2AncestralDiscreteScheduler = 11
    DEISMultistepScheduler = 12
    UniPCMultistepScheduler = 13
    DPMSolverSDEScheduler = 14
    EDMEulerScheduler = 15

AysSchedules = {
    "StableDiffusionTimesteps": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24],
    "StableDiffusionSigmas": [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.0],
    "StableDiffusionXLTimesteps": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13],
    "StableDiffusionXLSigmas": [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0],
    "StableDiffusionVideoSigmas": [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.0],
}

@dataclass
class SchedulerOutput(BaseOutput):
    class KarrasDiffusionSchedulers(Enum):
    DDIMScheduler = 1
    DDPMScheduler = 2
    PNDMScheduler = 3
    LMSDiscreteScheduler = 4
    EulerDiscreteScheduler = 5
    HeunDiscreteScheduler = 6
    EulerAncestralDiscreteScheduler = 7
    DPMSolverMultistepScheduler = 8
    DPMSolverSinglestepScheduler = 9
    KDPM2DiscreteScheduler = 10
    KDPM2AncestralDiscreteScheduler = 11
    DEISMultistepScheduler = 12
    UniPCMultistepScheduler = 13
    DPMSolverSDEScheduler = 14
    EDMEulerScheduler = 15

AysSchedules = {
    "StableDiffusionTimesteps": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24],
    "StableDiffusionSigmas": [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.0],
    "StableDiffusionXLTimesteps": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13],
    "StableDiffusionXLSigmas": [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0],
    "StableDiffusionVideoSigmas": [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.0],
}

@dataclass
class SchedulerOutput(BaseOutput):
    
    class KarrasDiffusionSchedulers(Enum):
    DDIMScheduler = 1
    DDPMScheduler = 2
    PNDMScheduler = 3
    LMSDiscreteScheduler = 4
    EulerDiscreteScheduler = 5
    HeunDiscreteScheduler = 6
    EulerAncestralDiscreteScheduler = 7
    DPMSolverMultistepScheduler = 8
    DPMSolverSinglestepScheduler = 9
    KDPM2DiscreteScheduler = 10
    KDPM2AncestralDiscreteScheduler = 11
    DEISMultistepScheduler = 12
    UniPCMultistepScheduler = 13
    DPMSolverSDEScheduler = 14
    EDMEulerScheduler = 15

AysSchedules = {
    "StableDiffusionTimesteps": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24],
    "StableDiffusionSigmas": [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.0],
    "StableDiffusionXLTimesteps": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13],
    "StableDiffusionXLSigmas": [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0],
    "StableDiffusionVideoSigmas": [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.0],
}

@dataclass
class SchedulerOutput(BaseOutput):
    class KarrasDiffusionSchedulers(Enum):
    DDIMScheduler = 1
    DDPMScheduler = 2
    PNDMScheduler = 3
    LMSDiscreteScheduler = 4
    EulerDiscreteScheduler = 5
    HeunDiscreteScheduler = 6
    EulerAncestralDiscreteScheduler = 7
    DPMSolverMultistepScheduler = 8
    DPMSolverSinglestepScheduler = 9
    KDPM2DiscreteScheduler = 10
    KDPM2AncestralDiscreteScheduler = 11
    DEISMultistepScheduler = 12
    UniPCMultistepScheduler = 13
    DPMSolverSDEScheduler = 14
    EDMEulerScheduler = 15

AysSchedules = {
    "StableDiffusionTimesteps": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24],
    "StableDiffusionSigmas": [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.0],
    "StableDiffusionXLTimesteps": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13],
    "StableDiffusionXLSigmas": [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0],
    "StableDiffusionVideoSigmas": [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.0],
}

@dataclass
class SchedulerOutput(BaseOutput):


    prev_sample: torch.Tensor

class SchedulerMixin(PushToHubMixin):
    
    class SchedulerMixin(PushToHubMixin):


    config_name = SCHEDULER_CONFIG_NAME
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
    ) -> Self:

        config, kwargs, commit_hash = cls.load_config(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            return_commit_hash=True,
            **kwargs,
        )
        return cls.from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        class should be returned or not.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.

        > [!TIP] > To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in
        with `hf > auth login`. You can also activate the special >
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a >
        firewalled environment.

        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)

    @property
    def compatibles(self):
        Returns all schedulers that are compatible with this scheduler
