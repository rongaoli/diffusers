import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import ByT5Tokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, T5EncoderModel

from ...guiders import AdaptiveProjectedMixGuidance
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLHunyuanImage, HunyuanImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import HunyuanImagePipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
        使用示例见文档

    Extract text enclosed in quotes for glyph rendering.

    Finds text in single quotes, double quotes, and Chinese quotes, then formats it for byT5 processing.

    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.
