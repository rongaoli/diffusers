from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput

@dataclass
class SkyReelsV2PipelineOutput(BaseOutput):


    frames: torch.Tensor
