from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin

class CLIPImageProjection(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.project = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.project(x)
