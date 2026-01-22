from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from ...utils import BaseOutput, is_flax_available

@dataclass
class StableDiffusionXLPipelineOutput(BaseOutput):


    images: Union[List[PIL.Image.Image], np.ndarray]

if is_flax_available():
    import flax

    @flax.struct.dataclass
    class FlaxStableDiffusionXLPipelineOutput(BaseOutput):


        images: np.ndarray
