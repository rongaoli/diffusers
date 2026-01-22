import inspect
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from transformers import (
    ByT5Tokenizer,
    Qwen2_5_VLTextModel,
    Qwen2Tokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
)

from ...guiders import ClassifierFreeGuidance
from ...models import AutoencoderKLHunyuanVideo15, HunyuanVideo15Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .image_processor import HunyuanVideo15ImageProcessor
from .pipeline_output import HunyuanVideo15PipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """


# Copied from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5.format_text_input
def format_text_input(prompt: List[str], system_message: str) -> List[Dict[str, Any]]:


    template = [
        [{"role": "system", "content": system_message}, {"role": "user", "content": p if p else " "}] for p in prompt
    ]

    return template

# Copied from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5.extract_glyph_texts
def extract_glyph_texts(prompt: str) -> List[str]:

    pattern = r"\"(.*?)\"|“(.*?)”"
    matches = re.findall(pattern, prompt)
    result = [match[0] or match[1] for match in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result

    if result:
        formatted_result = ". ".join([f'Text "{text}"' for text in result]) + ". "
    else:
        formatted_result = None

    return formatted_result

# Copied from diffusers.pipelines.stable_diffusion...
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

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
