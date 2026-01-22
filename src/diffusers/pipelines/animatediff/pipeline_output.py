from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image
import torch

from ...utils import BaseOutput

@dataclass
class AnimateDiffPipelineOutput(BaseOutput):


    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]
