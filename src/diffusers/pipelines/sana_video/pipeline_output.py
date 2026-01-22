from dataclasses import dataclass

import torch

from ...utils import BaseOutput

@dataclass
class SanaVideoPipelineOutput(BaseOutput):


    frames: torch.Tensor
