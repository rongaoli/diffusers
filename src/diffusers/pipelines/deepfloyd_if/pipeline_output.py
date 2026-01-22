from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL.Image

from ...utils import BaseOutput

@dataclass
class IFPipelineOutput(BaseOutput):


    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_detected: Optional[List[bool]]
    watermark_detected: Optional[List[bool]]
