import inspect
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from ...image_processor import VaeImageProcessor
from ...loaders import FromSingleFileMixin
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import LongCatImageTransformer2DModel
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from .pipeline_output import LongCatImagePipelineOutput
from .system_messages import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_ZH

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
        使用示例见文档

    Implement a regex-based string splitting algorithm that identifies delimiters defined by single or double quote
    pairs. Examples::
        >>> prompt_en = "Please write 'Hello' on the blackboard for me." >>> print(split_quotation(prompt_en)) >>> #
        output: [('Please write ', False), ("'Hello'", True), (' on the blackboard for me.', False)]

    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.
