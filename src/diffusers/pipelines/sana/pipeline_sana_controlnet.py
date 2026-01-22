import html
import inspect
import re
import urllib.parse as ul
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import Gemma2PreTrainedModel, GemmaTokenizer, GemmaTokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput, PixArtImageProcessor
from ...loaders import SanaLoraLoaderMixin
from ...models import AutoencoderDC, SanaControlNetModel, SanaTransformer2DModel
from ...schedulers import DPMSolverMultistepScheduler
from ...utils import (
    BACKENDS_MAPPING,
    USE_PEFT_BACKEND,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import get_device, is_torch_version, randn_tensor
from ..pipeline_utils import DiffusionPipeline
from ..pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)
from ..pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from .pipeline_output import SanaPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

ASPECT_RATIO_4096_BIN = {
    "0.25": [2048.0, 8192.0],
    "0.26": [2048.0, 7936.0],
    "0.27": [2048.0, 7680.0],
    "0.28": [2048.0, 7424.0],
    "0.32": [2304.0, 7168.0],
    "0.33": [2304.0, 6912.0],
    "0.35": [2304.0, 6656.0],
    "0.4": [2560.0, 6400.0],
    "0.42": [2560.0, 6144.0],
    "0.48": [2816.0, 5888.0],
    "0.5": [2816.0, 5632.0],
    "0.52": [2816.0, 5376.0],
    "0.57": [3072.0, 5376.0],
    "0.6": [3072.0, 5120.0],
    "0.68": [3328.0, 4864.0],
    "0.72": [3328.0, 4608.0],
    "0.78": [3584.0, 4608.0],
    "0.82": [3584.0, 4352.0],
    "0.88": [3840.0, 4352.0],
    "0.94": [3840.0, 4096.0],
    "1.0": [4096.0, 4096.0],
    "1.07": [4096.0, 3840.0],
    "1.13": [4352.0, 3840.0],
    "1.21": [4352.0, 3584.0],
    "1.29": [4608.0, 3584.0],
    "1.38": [4608.0, 3328.0],
    "1.46": [4864.0, 3328.0],
    "1.67": [5120.0, 3072.0],
    "1.75": [5376.0, 3072.0],
    "2.0": [5632.0, 2816.0],
    "2.09": [5888.0, 2816.0],
    "2.4": [6144.0, 2560.0],
    "2.5": [6400.0, 2560.0],
    "2.89": [6656.0, 2304.0],
    "3.0": [6912.0, 2304.0],
    "3.11": [7168.0, 2304.0],
    "3.62": [7424.0, 2048.0],
    "3.75": [7680.0, 2048.0],
    "3.88": [7936.0, 2048.0],
    "4.0": [8192.0, 2048.0],
}

EXAMPLE_DOC_STRING = """
        使用示例见文档

    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.
