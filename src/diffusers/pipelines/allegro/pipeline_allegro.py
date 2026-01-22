import html
import inspect
import math
import re
import urllib.parse as ul
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...models import AllegroTransformer3DModel, AutoencoderKLAllegro
from ...models.embeddings import get_3d_rotary_pos_embed_allegro
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from .pipeline_output import AllegroPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

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
