from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL.Image

from ...utils import BaseOutput, is_flax_available

@dataclass
class StableDiffusionPipelineOutput(BaseOutput):


    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]

if is_flax_available():
    import flax

    @flax.struct.dataclass
    class FlaxStableDiffusionPipelineOutput(BaseOutput):


        images: np.ndarray
        nsfw_content_detected: List[bool]
