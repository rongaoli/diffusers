import inspect
from typing import List, Optional, Tuple, Union

import torch

from ...models import WanTransformer3DModel
from ...schedulers import UniPCMultistepScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import WanModularPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# TODO(yiyi, aryan): We need another step before text encoder to set the `num_inference_steps` attribute for guider so that
# things like when to do guidance and how many con...
# always assuming you want to do guidance in the G...
# configuration of guider is.

def repeat_tensor_to_batch_size(
    input_name: str,
    input_tensor: torch.Tensor,
    batch_size: int,
    num_videos_per_prompt: int = 1,
) -> torch.Tensor:

    # make sure input is a tensor
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError(f"`{input_name}` must be a tensor")

    # make sure input tensor e.g. image_latents has batch size 1 or batch_size same as prompts
    if input_tensor.shape[0] == 1:
        repeat_by = batch_size * num_videos_per_prompt
    elif input_tensor.shape[0] == batch_size:
        repeat_by = num_videos_per_prompt
    else:
        raise ValueError(
            f"`{input_name}` must have have batch size 1 or {batch_size}, but got {input_tensor.shape[0]}"
        )

    # expand the tensor to match the batch_size * num_videos_per_prompt
    input_tensor = input_tensor.repeat_interleave(repeat_by, dim=0)

    return input_tensor

def calculate_dimension_from_latents(
    latents: torch.Tensor, vae_scale_factor_temporal: int, vae_scale_factor_spatial: int
) -> Tuple[int, int]:

    if latents.ndim != 5:
        raise ValueError(f"latents must have 5 dimensions, but got {latents.ndim}")

    _, _, num_latent_frames, latent_height, latent_width = latents.shape

    num_frames = (num_latent_frames - 1) * vae_scale_factor_temporal + 1
    height = latent_height * vae_scale_factor_spatial
    width = latent_width * vae_scale_factor_spatial

    return num_frames, height, width

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):

    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.
