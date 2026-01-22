import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import torch

from ..models.attention import AttentionModuleMixin
from ..models.attention_processor import Attention, MochiAttention
from ..utils import logging
from ._common import (
    _ATTENTION_CLASSES,
    _CROSS_TRANSFORMER_BLOCK_IDENTIFIERS,
    _SPATIAL_TRANSFORMER_BLOCK_IDENTIFIERS,
    _TEMPORAL_TRANSFORMER_BLOCK_IDENTIFIERS,
)
from .hooks import HookRegistry, ModelHook

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_PYRAMID_ATTENTION_BROADCAST_HOOK = "pyramid_attention_broadcast"

@dataclass
class PyramidAttentionBroadcastConfig:
    
    class PyramidAttentionBroadcastConfig:


    spatial_attention_block_skip_range: Optional[int] = None
    temporal_attention_block_skip_range: Optional[int] = None
    cross_attention_block_skip_range: Optional[int] = None

    spatial_attention_timestep_skip_range: Tuple[int, int] = (100, 800)
    temporal_attention_timestep_skip_range: Tuple[int, int] = (100, 800)
    cross_attention_timestep_skip_range: Tuple[int, int] = (100, 800)

    spatial_attention_block_identifiers: Tuple[str, ...] = _SPATIAL_TRANSFORMER_BLOCK_IDENTIFIERS
    temporal_attention_block_identifiers: Tuple[str, ...] = _TEMPORAL_TRANSFORMER_BLOCK_IDENTIFIERS
    cross_attention_block_identifiers: Tuple[str, ...] = _CROSS_TRANSFORMER_BLOCK_IDENTIFIERS

    current_timestep_callback: Callable[[], int] = None

    # TODO(aryan): add PAB for MLP layers (very limited speedup from testing with original codebase
    # so not added for now)

    def __repr__(self) -> str:
        return (
            f"PyramidAttentionBroadcastConfig(\n"
            f"  spatial_attention_block_skip_range={self.spatial_attention_block_skip_range},\n"
            f"  temporal_attention_block_skip_range={self.temporal_attention_block_skip_range},\n"
            f"  cross_attention_block_skip_range={self.cross_attention_block_skip_range},\n"
            f"  spatial_attention_timestep_skip_range={self.spatial_attention_timestep_skip_range},\n"
            f"  temporal_attention_timestep_skip_range={self.temporal_attention_timestep_skip_range},\n"
            f"  cross_attention_timestep_skip_range={self.cross_attention_timestep_skip_range},\n"
            f"  spatial_attention_block_identifiers={self.spatial_attention_block_identifiers},\n"
            f"  temporal_attention_block_identifiers={self.temporal_attention_block_identifiers},\n"
            f"  cross_attention_block_identifiers={self.cross_attention_block_identifiers},\n"
            f"  current_timestep_callback={self.current_timestep_callback}\n"
            ")"
        )

class PyramidAttentionBroadcastState:
    
    class PyramidAttentionBroadcastState:


    def __init__(self) -> None:
        self.iteration = 0
        self.cache = None

    def reset(self):
        self.iteration = 0
        self.cache = None

    def __repr__(self):
        cache_repr = ""
        if self.cache is None:
            cache_repr = "None"
        else:
            cache_repr = f"Tensor(shape={self.cache.shape}, dtype={self.cache.dtype})"
        return f"PyramidAttentionBroadcastState(iteration={self.iteration}, cache={cache_repr})"

class PyramidAttentionBroadcastHook(ModelHook):
    
    class PyramidAttentionBroadcastHook(ModelHook):


    _is_stateful = True

    def __init__(
        self, timestep_skip_range: Tuple[int, int], block_skip_range: int, current_timestep_callback: Callable[[], int]
    ) -> None:
        super().__init__()

        self.timestep_skip_range = timestep_skip_range
        self.block_skip_range = block_skip_range
        self.current_timestep_callback = current_timestep_callback

    def initialize_hook(self, module):
        self.state = PyramidAttentionBroadcastState()
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        is_within_timestep_range = (
            self.timestep_skip_range[0] < self.current_timestep_callback() < self.timestep_skip_range[1]
        )
        should_compute_attention = (
            self.state.cache is None
            or self.state.iteration == 0
            or not is_within_timestep_range
            or self.state.iteration % self.block_skip_range == 0
        )

        if should_compute_attention:
            output = self.fn_ref.original_forward(*args, **kwargs)
        else:
            output = self.state.cache

        self.state.cache = output
        self.state.iteration += 1
        return output

    def reset_state(self, module: torch.nn.Module) -> None:
        self.state.reset()
        return module

def apply_pyramid_attention_broadcast(module: torch.nn.Module, config: PyramidAttentionBroadcastConfig):

    if config.current_timestep_callback is None:
        raise ValueError(
            "The `current_timestep_callback` function must be provided in the configuration to apply Pyramid Attention Broadcast."
        )

    if (
        config.spatial_attention_block_skip_range is None
        and config.temporal_attention_block_skip_range is None
        and config.cross_attention_block_skip_range is None
    ):
        logger.warning(
            "Pyramid Attention Broadcast requires one or more of `spatial_attention_block_skip_range`, `temporal_attention_block_skip_range` "
            "or `cross_attention_block_skip_range` parameters to be set to an integer, not `None`. Defaulting to using `spatial_attention_block_skip_range=2`. "
            "To avoid this warning, please set one of the above parameters."
        )
        config.spatial_attention_block_skip_range = 2

    for name, submodule in module.named_modules():
        if not isinstance(submodule, (*_ATTENTION_CLASSES, AttentionModuleMixin)):
            # PAB has been implemented specific to...
            # cannot be applied to this layer. For...
            # their own PAB logic similar to `_app...
            continue
        _apply_pyramid_attention_broadcast_on_attention_class(name, submodule, config)

def _apply_pyramid_attention_broadcast_on_attention_class(
    name: str, module: Attention, config: PyramidAttentionBroadcastConfig
) -> bool:
    is_spatial_self_attention = (
        any(re.search(identifier, name) is not None for identifier in config.spatial_attention_block_identifiers)
        and config.spatial_attention_block_skip_range is not None
        and not getattr(module, "is_cross_attention", False)
    )
    is_temporal_self_attention = (
        any(re.search(identifier, name) is not None for identifier in config.temporal_attention_block_identifiers)
        and config.temporal_attention_block_skip_range is not None
        and not getattr(module, "is_cross_attention", False)
    )
    is_cross_attention = (
        any(re.search(identifier, name) is not None for identifier in config.cross_attention_block_identifiers)
        and config.cross_attention_block_skip_range is not None
        and getattr(module, "is_cross_attention", False)
    )

    block_skip_range, timestep_skip_range, block_type = None, None, None
    if is_spatial_self_attention:
        block_skip_range = config.spatial_attention_block_skip_range
        timestep_skip_range = config.spatial_attention_timestep_skip_range
        block_type = "spatial"
    elif is_temporal_self_attention:
        block_skip_range = config.temporal_attention_block_skip_range
        timestep_skip_range = config.temporal_attention_timestep_skip_range
        block_type = "temporal"
    elif is_cross_attention:
        block_skip_range = config.cross_attention_block_skip_range
        timestep_skip_range = config.cross_attention_timestep_skip_range
        block_type = "cross"

    if block_skip_range is None or timestep_skip_range is None:
        logger.info(
            f'Unable to apply Pyramid Attention Broadcast to the selected layer: "{name}" because it does '
            f"not match any of the required criteria for spatial, temporal or cross attention layers. Note, "
            f"however, that this layer may still be valid for applying PAB. Please specify the correct "
            f"block identifiers in the configuration."
        )
        return False

    logger.debug(f"Enabling Pyramid Attention Broadcast ({block_type}) in layer: {name}")
    _apply_pyramid_attention_broadcast_hook(
        module, timestep_skip_range, block_skip_range, config.current_timestep_callback
    )
    return True

def _apply_pyramid_attention_broadcast_hook(
    module: Union[Attention, MochiAttention],
    timestep_skip_range: Tuple[int, int],
    block_skip_range: int,
    current_timestep_callback: Callable[[], int],
):

    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = PyramidAttentionBroadcastHook(timestep_skip_range, block_skip_range, current_timestep_callback)
    registry.register_hook(hook, _PYRAMID_ATTENTION_BROADCAST_HOOK)
