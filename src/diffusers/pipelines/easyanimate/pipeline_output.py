from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput

@dataclass
class EasyAnimatePipelineOutput(BaseOutput):


    frames: torch.Tensor
