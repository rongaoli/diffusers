import re
from typing import Optional, Tuple, Type, Union

import torch

from ..utils import get_logger, is_peft_available, is_peft_version
from ._common import _GO_LC_SUPPORTED_PYTORCH_LAYERS
from .hooks import HookRegistry, ModelHook

logger = get_logger(__name__)  # pylint: disable=invalid-name

# fmt: off
_LAYERWISE_CASTING_HOOK = "layerwise_casting"
_PEFT_AUTOCAST_DISABLE_HOOK = "peft_autocast_disable"
DEFAULT_SKIP_MODULES_PATTERN = ("pos_embed", "patch_embed", "norm", "^proj_in$", "^proj_out$")
# fmt: on

_SHOULD_DISABLE_PEFT_INPUT_AUTOCAST = is_peft_available() and is_peft_version(">", "0.14.0")
if _SHOULD_DISABLE_PEFT_INPUT_AUTOCAST:
    from peft.helpers import disable_input_dtype_casting
    from peft.tuners.tuners_utils import BaseTunerLayer

class LayerwiseCastingHook(ModelHook):
    
    class LayerwiseCastingHook(ModelHook):


    _is_stateful = False

    def __init__(self, storage_dtype: torch.dtype, compute_dtype: torch.dtype, non_blocking: bool) -> None:
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype
        self.non_blocking = non_blocking

    def initialize_hook(self, module: torch.nn.Module):
        module.to(dtype=self.storage_dtype, non_blocking=self.non_blocking)
        return module

    def deinitalize_hook(self, module: torch.nn.Module):
        raise NotImplementedError(
            "LayerwiseCastingHook does not support deinitialization. A model once enabled with layerwise casting will "
            "have casted its weights to a lower precision dtype for storage. Casting this back to the original dtype "
            "will lead to precision loss, which might have an impact on the model's generation quality. The model should "
            "be re-initialized and loaded in the original dtype."
        )

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        module.to(dtype=self.compute_dtype, non_blocking=self.non_blocking)
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output):
        module.to(dtype=self.storage_dtype, non_blocking=self.non_blocking)
        return output

class PeftInputAutocastDisableHook(ModelHook):
    
    class PeftInputAutocastDisableHook(ModelHook):


    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        with disable_input_dtype_casting(module):
            return self.fn_ref.original_forward(*args, **kwargs)

def apply_layerwise_casting(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: Union[str, Tuple[str, ...]] = "auto",
    skip_modules_classes: Optional[Tuple[Type[torch.nn.Module], ...]] = None,
    non_blocking: bool = False,
) -> None:

    if skip_modules_pattern == "auto":
        skip_modules_pattern = DEFAULT_SKIP_MODULES_PATTERN

    if skip_modules_classes is None and skip_modules_pattern is None:
        apply_layerwise_casting_hook(module, storage_dtype, compute_dtype, non_blocking)
        return

    _apply_layerwise_casting(
        module,
        storage_dtype,
        compute_dtype,
        skip_modules_pattern,
        skip_modules_classes,
        non_blocking,
    )
    _disable_peft_input_autocast(module)

def _apply_layerwise_casting(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: Optional[Tuple[str, ...]] = None,
    skip_modules_classes: Optional[Tuple[Type[torch.nn.Module], ...]] = None,
    non_blocking: bool = False,
    _prefix: str = "",
) -> None:
    should_skip = (skip_modules_classes is not None and isinstance(module, skip_modules_classes)) or (
        skip_modules_pattern is not None and any(re.search(pattern, _prefix) for pattern in skip_modules_pattern)
    )
    if should_skip:
        logger.debug(f'Skipping layerwise casting for layer "{_prefix}"')
        return

    if isinstance(module, _GO_LC_SUPPORTED_PYTORCH_LAYERS):
        logger.debug(f'Applying layerwise casting to layer "{_prefix}"')
        apply_layerwise_casting_hook(module, storage_dtype, compute_dtype, non_blocking)
        return

    for name, submodule in module.named_children():
        layer_name = f"{_prefix}.{name}" if _prefix else name
        _apply_layerwise_casting(
            submodule,
            storage_dtype,
            compute_dtype,
            skip_modules_pattern,
            skip_modules_classes,
            non_blocking,
            _prefix=layer_name,
        )

def apply_layerwise_casting_hook(
    module: torch.nn.Module, storage_dtype: torch.dtype, compute_dtype: torch.dtype, non_blocking: bool
) -> None:

    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = LayerwiseCastingHook(storage_dtype, compute_dtype, non_blocking)
    registry.register_hook(hook, _LAYERWISE_CASTING_HOOK)

def _is_layerwise_casting_active(module: torch.nn.Module) -> bool:
    for submodule in module.modules():
        if (
            hasattr(submodule, "_diffusers_hook")
            and submodule._diffusers_hook.get_hook(_LAYERWISE_CASTING_HOOK) is not None
        ):
            return True
    return False

def _disable_peft_input_autocast(module: torch.nn.Module) -> None:
    if not _SHOULD_DISABLE_PEFT_INPUT_AUTOCAST:
        return
    for submodule in module.modules():
        if isinstance(submodule, BaseTunerLayer) and _is_layerwise_casting_active(submodule):
            registry = HookRegistry.check_if_exists_or_initialize(submodule)
            hook = PeftInputAutocastDisableHook()
            registry.register_hook(hook, _PEFT_AUTOCAST_DISABLE_HOOK)
