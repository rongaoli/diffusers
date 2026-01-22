from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image
import torch

from ...utils import BaseOutput

@dataclass
class FluxPipelineOutput(BaseOutput):


    images: Union[List[PIL.Image.Image], np.ndarray]

@dataclass
class FluxPriorReduxPipelineOutput(BaseOutput):


    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
