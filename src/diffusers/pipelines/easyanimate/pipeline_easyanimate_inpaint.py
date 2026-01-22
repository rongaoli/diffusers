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

def get_image_to_video_latent(validation_image_start, validation_image_end, num_frames, sample_size):

    input_video = None
    input_video_mask = None

    if validation_image_start is not None:
        # Preprocess the starting image(s)
        if isinstance(validation_image_start, list):
            image_start = [preprocess_image(img, sample_size) for img in validation_image_start]
        else:
            image_start = preprocess_image(validation_image_start, sample_size)

        # Create video tensor from the starting image(s)
        if isinstance(image_start, list):
            start_video = torch.cat(
                [img.unsqueeze(1).unsqueeze(0) for img in image_start],
                dim=2,
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, num_frames, 1, 1])
            input_video[:, :, : len(image_start)] = start_video
        else:
            input_video = torch.tile(
                image_start.unsqueeze(1).unsqueeze(0),
                [1, 1, num_frames, 1, 1],
            )

        # Normalize input video (already normalized in preprocess_image)

        # Create mask for the input video
        input_video_mask = torch.zeros_like(input_video[:, :1])
        if isinstance(image_start, list):
            input_video_mask[:, :, len(image_start) :] = 255
        else:
            input_video_mask[:, :, 1:] = 255

        # Handle ending image(s) if provided
        if validation_image_end is not None:
            if isinstance(validation_image_end, list):
                image_end = [preprocess_image(img, sample_size) for img in validation_image_end]
                end_video = torch.cat(
                    [img.unsqueeze(1).unsqueeze(0) for img in image_end],
                    dim=2,
                )
                input_video[:, :, -len(end_video) :] = end_video
                input_video_mask[:, :, -len(image_end) :] = 0
            else:
                image_end = preprocess_image(validation_image_end, sample_size)
                input_video[:, :, -1:] = image_end.unsqueeze(1).unsqueeze(0)
                input_video_mask[:, :, -1:] = 0

    elif validation_image_start is None:
        # If no starting image is provided, initialize empty tensors
        input_video = torch.zeros([1, 3, num_frames, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, num_frames, sample_size[0], sample_size[1]]) * 255

    return input_video, input_video_mask

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

## Add noise to reference video
def add_noise_to_reference_video(image, ratio=None, generator=None):
    if ratio is None:
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(image.device)
        sigma = torch.exp(sigma).to(image.dtype)
    else:
        sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio

    if generator is not None:
        image_noise = (
            torch.randn(image.size(), generator=generator, dtype=image.dtype, device=image.device)
            * sigma[:, None, None, None, None]
        )
    else:
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
    image_noise = torch.where(image == -1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image

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
