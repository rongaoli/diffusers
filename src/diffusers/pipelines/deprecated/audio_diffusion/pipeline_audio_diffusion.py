from math import acos, sin
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ....models import AutoencoderKL, UNet2DConditionModel
from ....schedulers import DDIMScheduler, DDPMScheduler
from ....utils.torch_utils import randn_tensor
from ...pipeline_utils import AudioPipelineOutput, BaseOutput, DiffusionPipeline, ImagePipelineOutput
from .mel import Mel

class AudioDiffusionPipeline(DiffusionPipeline):


    _optional_components = ["vqvae"]

    def __init__(
        self,
        vqvae: AutoencoderKL,
        unet: UNet2DConditionModel,
        mel: Mel,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, mel=mel, vqvae=vqvae)

    def get_default_steps(self) -> int:
        """Returns default number of steps recommended for inference.
