from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL.Image

from ...utils import (
    BaseOutput,
)

@dataclass
class StableDiffusionSafePipelineOutput(BaseOutput):


    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    unsafe_images: Optional[Union[List[PIL.Image.Image], np.ndarray]]
    applied_safety_concept: Optional[str]
