from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL
import torch

from ...utils import (
    BaseOutput,
)

@dataclass
class TextToVideoSDPipelineOutput(BaseOutput):


    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]
