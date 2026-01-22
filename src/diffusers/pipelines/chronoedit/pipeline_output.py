from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput

@dataclass
class ChronoEditPipelineOutput(BaseOutput):


    frames: torch.Tensor
