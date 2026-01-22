from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput

@dataclass
class KandinskyPipelineOutput(BaseOutput):


    frames: torch.Tensor

@dataclass
class KandinskyImagePipelineOutput(BaseOutput):


    image: torch.Tensor
