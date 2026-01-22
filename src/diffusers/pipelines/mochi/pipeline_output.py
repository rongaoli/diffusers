from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput

@dataclass
class MochiPipelineOutput(BaseOutput):


    frames: torch.Tensor
