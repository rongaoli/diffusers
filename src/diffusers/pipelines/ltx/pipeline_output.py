from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput

@dataclass
class LTXPipelineOutput(BaseOutput):


    frames: torch.Tensor
