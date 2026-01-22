import html
import inspect
import re
import urllib.parse as ul
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import Gemma2PreTrainedModel, GemmaTokenizer, GemmaTokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import SanaLoraLoaderMixin
from ...models import AutoencoderDC, AutoencoderKLWan, SanaVideoTransformer3DModel
from ...schedulers import DPMSolverMultistepScheduler
from ...utils import (
    BACKENDS_MAPPING,
    USE_PEFT_BACKEND,
    is_bs4_available,
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import get_device, is_torch_version, randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import SanaVideoPipelineOutput

ASPECT_RATIO_480_BIN = {
    "0.5": [448.0, 896.0],
    "0.57": [480.0, 832.0],
    "0.68": [528.0, 768.0],
    "0.78": [560.0, 720.0],
    "1.0": [624.0, 624.0],
    "1.13": [672.0, 592.0],
    "1.29": [720.0, 560.0],
    "1.46": [768.0, 528.0],
    "1.67": [816.0, 496.0],
    "1.75": [832.0, 480.0],
    "2.0": [896.0, 448.0],
}

ASPECT_RATIO_720_BIN = {
    "0.5": [672.0, 1344.0],
    "0.57": [704.0, 1280.0],
    "0.68": [800.0, 1152.0],
    "0.78": [832.0, 1088.0],
    "1.0": [960.0, 960.0],
    "1.13": [1024.0, 896.0],
    "1.29": [1088.0, 832.0],
    "1.46": [1152.0, 800.0],
    "1.67": [1248.0, 736.0],
    "1.75": [1280.0, 704.0],
    "2.0": [1344.0, 672.0],
}

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
