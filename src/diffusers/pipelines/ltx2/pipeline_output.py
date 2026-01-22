from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput

@dataclass
class LTX2PipelineOutput(BaseOutput):


    frames: torch.Tensor
    audio: torch.Tensor
