from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput

@dataclass
class CogVideoXPipelineOutput(BaseOutput):


    frames: torch.Tensor
