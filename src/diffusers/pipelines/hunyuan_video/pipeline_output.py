from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image
import torch

from diffusers.utils import BaseOutput

@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):


    frames: torch.Tensor

@dataclass
class HunyuanVideoFramepackPipelineOutput(BaseOutput):


    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]], List[torch.Tensor]]
