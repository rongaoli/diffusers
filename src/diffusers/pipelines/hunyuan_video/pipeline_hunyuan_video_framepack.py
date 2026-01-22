import inspect
import math
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    LlamaModel,
    LlamaTokenizerFast,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import HunyuanVideoLoraLoaderMixin
from ...models import AutoencoderKLHunyuanVideo, HunyuanVideoFramepackTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import HunyuanVideoFramepackPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# TODO(yiyi): We can pack the checkpoints nicely with modular loader
EXAMPLE_DOC_STRING = """
    Examples:
        ##### Image-to-Video

        ```python
        >>> import torch
        >>> from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
        >>> from diffusers.utils import export_to_video, load_image
        >>> from transformers import SiglipImageProcessor, SiglipVisionModel

        >>> transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
        ...     "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
        ... )
        >>> feature_extractor = SiglipImageProcessor.from_pretrained(
        ...     "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
        ... )
        >>> image_encoder = SiglipVisionModel.from_pretrained(
        ...     "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
        ... )
        >>> pipe = HunyuanVideoFramepackPipeline.from_pretrained(
        ...     "hunyuanvideo-community/HunyuanVideo",
        ...     transformer=transformer,
        ...     feature_extractor=feature_extractor,
        ...     image_encoder=image_encoder,
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe.vae.enable_tiling()
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
        ... )
        >>> output = pipe(
        ...     image=image,
        ...     prompt="A penguin dancing in the snow",
        ...     height=832,
        ...     width=480,
        ...     num_frames=91,
        ...     num_inference_steps=30,
        ...     guidance_scale=9.0,
        ...     generator=torch.Generator().manual_seed(0),
        ...     sampling_type="inverted_anti_drifting",
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=30)
        ```

        ##### First and Last Image-to-Video

        ```python
        >>> import torch
        >>> from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
        >>> from diffusers.utils import export_to_video, load_image
        >>> from transformers import SiglipImageProcessor, SiglipVisionModel

        >>> transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
        ...     "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
        ... )
        >>> feature_extractor = SiglipImageProcessor.from_pretrained(
        ...     "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
        ... )
        >>> image_encoder = SiglipVisionModel.from_pretrained(
        ...     "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
        ... )
        >>> pipe = HunyuanVideoFramepackPipeline.from_pretrained(
        ...     "hunyuanvideo-community/HunyuanVideo",
        ...     transformer=transformer,
        ...     feature_extractor=feature_extractor,
        ...     image_encoder=image_encoder,
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe.to("cuda")

        >>> prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
        >>> first_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
        ... )
        >>> last_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
        ... )
        >>> output = pipe(
        ...     image=first_image,
        ...     last_image=last_image,
        ...     prompt=prompt,
        ...     height=512,
        ...     width=512,
        ...     num_frames=91,
        ...     num_inference_steps=30,
        ...     guidance_scale=9.0,
        ...     generator=torch.Generator().manual_seed(0),
        ...     sampling_type="inverted_anti_drifting",
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=30)
        ```

DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

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
