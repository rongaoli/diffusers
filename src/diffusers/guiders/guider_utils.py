import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub.utils import validate_hf_hub_args
from typing_extensions import Self

from ..configuration_utils import ConfigMixin
from ..utils import BaseOutput, PushToHubMixin, get_logger

if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState

GUIDER_CONFIG_NAME = "guider_config.json"

logger = get_logger(__name__)  # pylint: disable=invalid-name

class BaseGuidance(ConfigMixin, PushToHubMixin):
    
    class BaseGuidance(ConfigMixin, PushToHubMixin):


    config_name = GUIDER_CONFIG_NAME
    _input_predictions = None
    _identifier_key = "__guidance_identifier__"

    def __init__(self, start: float = 0.0, stop: float = 1.0, enabled: bool = True):
        logger.warning(
            "Guiders are currently an experimental feature under active development. The API is subject to breaking changes in future releases."
        )

        self._start = start
        self._stop = stop
        self._step: int = None
        self._num_inference_steps: int = None
        self._timestep: torch.LongTensor = None
        self._count_prepared = 0
        self._input_fields: Dict[str, Union[str, Tuple[str, str]]] = None
        self._enabled = enabled

        if not (0.0 <= start < 1.0):
            raise ValueError(f"Expected `start` to be between 0.0 and 1.0, but got {start}.")
        if not (start <= stop <= 1.0):
            raise ValueError(f"Expected `stop` to be between {start} and 1.0, but got {stop}.")

        if self._input_predictions is None or not isinstance(self._input_predictions, list):
            raise ValueError(
                "`_input_predictions` must be a list of required prediction names for the guidance technique."
            )

    def new(self, **kwargs):
        Creates a copy of this guider instance, optionally with modified configuration parameters.
            **kwargs: Configuration parameters to override in the new instance. If no kwargs are provided,
                returns an exact copy with the same configuration.
