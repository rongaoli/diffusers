from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput

@dataclass
class LucyPipelineOutput(BaseOutput):


    frames: torch.Tensor
