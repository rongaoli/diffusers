from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..models.attention import BasicTransformerBlock, FreeNoiseTransformerBlock
from ..models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from ..models.transformers.transformer_2d import Transformer2DModel
from ..models.unets.unet_motion_model import (
    AnimateDiffTransformer3D,
    CrossAttnDownBlockMotion,
    DownBlockMotion,
    UpBlockMotion,
)
from ..pipelines.pipeline_utils import DiffusionPipeline
from ..utils import logging
from ..utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class SplitInferenceModule(nn.Module):


    def __init__(
        self,
        module: nn.Module,
        split_size: int = 1,
        split_dim: int = 0,
        input_kwargs_to_split: List[str] = ["hidden_states"],
    ) -> None:
        super().__init__()

        self.module = module
        self.split_size = split_size
        self.split_dim = split_dim
        self.input_kwargs_to_split = set(input_kwargs_to_split)

    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        r"""Forward method for the `SplitInferenceModule`.

        This method processes the input by splitting specified keyword arguments along a given dimension, running the
        underlying module on each split, and then concatenating the results. The splitting is controlled by the
        `split_size` and `split_dim` parameters specified during initialization.
            *args (`Any`):
                Positional arguments that are passed directly to the `module` without modification.
            **kwargs (`Dict[str, torch.Tensor]`):
                Keyword arguments passed to the underlying `module`. Only keyword arguments whose names match the
                entries in `input_kwargs_to_split` and are of type `torch.Tensor` will be split. The remaining keyword
                arguments are passed unchanged.
