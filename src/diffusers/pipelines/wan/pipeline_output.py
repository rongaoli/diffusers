from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput

@dataclass
class WanPipelineOutput(BaseOutput):


    frames: torch.Tensor
