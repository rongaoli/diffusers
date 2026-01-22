from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image
import torch

from diffusers.utils import BaseOutput, get_logger

logger = get_logger(__name__)

@dataclass
class CosmosPipelineOutput(BaseOutput):


    frames: torch.Tensor

@dataclass
class CosmosImagePipelineOutput(BaseOutput):


    images: Union[List[PIL.Image.Image], np.ndarray]
