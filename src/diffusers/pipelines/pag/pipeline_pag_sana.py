import html
import inspect
import re
import urllib.parse as ul
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import Gemma2PreTrainedModel, GemmaTokenizer, GemmaTokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PixArtImageProcessor
from ...models import AutoencoderDC, SanaTransformer2DModel
from ...models.attention_processor import PAGCFGSanaLinearAttnProcessor2_0, PAGIdentitySanaLinearAttnProcessor2_0
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import get_device, is_torch_version, randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ..pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)
from ..pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from ..sana.pipeline_sana import ASPECT_RATIO_4096_BIN
from .pag_utils import PAGMixin

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

EXAMPLE_DOC_STRING = """
        使用示例见文档

    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.
