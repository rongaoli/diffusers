from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin
from ...utils import BaseOutput

@dataclass
class ReduxImageEncoderOutput(BaseOutput):
    image_embeds: Optional[torch.Tensor] = None

class ReduxImageEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
    ) -> None:
        super().__init__()

        self.redux_up = nn.Linear(redux_dim, txt_in_features * 3)
        self.redux_down = nn.Linear(txt_in_features * 3, txt_in_features)

    def forward(self, x: torch.Tensor) -> ReduxImageEncoderOutput:
        projected_x = self.redux_down(nn.functional.silu(self.redux_up(x)))

        return ReduxImageEncoderOutput(image_embeds=projected_x)
