import inspect
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    BertModel,
    BertTokenizer,
    Qwen2Tokenizer,
    Qwen2VLForConditionalGeneration,
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLMagvit, EasyAnimateTransformer3DModel
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from .pipeline_output import EasyAnimatePipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """


def preprocess_image(image, sample_size):

    if isinstance(image, torch.Tensor):
        # If input is a tensor, assume it's in CHW format and resize using interpolation
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=sample_size, mode="bilinear", align_corners=False
        ).squeeze(0)
    elif isinstance(image, Image.Image):
        # If input is a PIL image, resize and convert to numpy array
        image = image.resize((sample_size[1], sample_size[0]))
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        # If input is a numpy array, resize using PIL
        image = Image.fromarray(image).resize((sample_size[1], sample_size[0]))
        image = np.array(image)
    else:
        raise ValueError("Unsupported input type. Expected PIL.Image, numpy.ndarray, or torch.Tensor.")

    # Convert to tensor if not already
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # HWC -> CHW, normalize to [0, 1]

    return image

def get_video_to_video_latent(input_video, num_frames, sample_size, validation_video_mask=None, ref_image=None):
    if input_video is not None:
        # Convert each frame in the list to tensor
        input_video = [preprocess_image(frame, sample_size=sample_size) for frame in input_video]

        # Stack all frames into a single tensor (F, C, H, W)
        input_video = torch.stack(input_video)[:num_frames]

        # Add batch dimension (B, F, C, H, W)
        input_video = input_video.permute(1, 0, 2, 3).unsqueeze(0)

        if validation_video_mask is not None:
            # Handle mask input
            validation_video_mask = preprocess_image(validation_video_mask, size=sample_size)
            input_video_mask = torch.where(validation_video_mask < 240 / 255.0, 0.0, 255)

            # Adjust mask dimensions to match video
            input_video_mask = input_video_mask.unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
            input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
            input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
        else:
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, :] = 255
    else:
        input_video, input_video_mask = None, None

    if ref_image is not None:
        # Convert reference image to tensor
        ref_image = preprocess_image(ref_image, size=sample_size)
        ref_image = ref_image.permute(1, 0, 2, 3).unsqueeze(0)  # Add batch dimension (B, C, H, W)
    else:
        ref_image = None

    return input_video, input_video_mask, ref_image

# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    
    """r"""
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance...
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

# Resize mask information in magvit
def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :], size=target_size, mode="trilinear", align_corners=False
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :], size=target_size, mode="trilinear", align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(mask, size=target_size, mode="trilinear", align_corners=False)
    return resized_mask

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
