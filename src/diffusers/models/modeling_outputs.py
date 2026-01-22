from dataclasses import dataclass

from ..utils import BaseOutput

@dataclass
class AutoencoderKLOutput(BaseOutput):


    latent_dist: "DiagonalGaussianDistribution"  # noqa: F821

@dataclass
class Transformer2DModelOutput(BaseOutput):


    sample: "torch.Tensor"  # noqa: F821
