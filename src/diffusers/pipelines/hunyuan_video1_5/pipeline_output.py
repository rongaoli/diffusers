from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput

@dataclass
class HunyuanVideo15PipelineOutput(BaseOutput):


    frames: torch.Tensor
