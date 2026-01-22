from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL.Image

from ...utils import BaseOutput

@dataclass
class SemanticStableDiffusionPipelineOutput(BaseOutput):


    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
