import contextlib
import functools
import inspect
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

if torch.distributed.is_available():
    import torch.distributed._functional_collectives as funcol

from ..utils import (
    get_logger,
    is_aiter_available,
    is_aiter_version,
    is_flash_attn_3_available,
    is_flash_attn_available,
    is_flash_attn_version,
    is_kernels_available,
    is_sageattention_available,
    is_sageattention_version,
    is_torch_npu_available,
    is_torch_version,
    is_torch_xla_available,
    is_torch_xla_version,
    is_xformers_available,
    is_xformers_version,
)
from ..utils.constants import DIFFUSERS_ATTN_BACKEND, DIFFUSERS_ATTN_CHECKS

if TYPE_CHECKING:
    from ._modeling_parallel import ParallelConfig

_REQUIRED_FLASH_VERSION = "2.6.3"
_REQUIRED_AITER_VERSION = "0.1.5"
_REQUIRED_SAGE_VERSION = "2.1.1"
_REQUIRED_FLEX_VERSION = "2.5.0"
_REQUIRED_XLA_VERSION = "2.2"
_REQUIRED_XFORMERS_VERSION = "0.0.29"

_CAN_USE_FLASH_ATTN = is_flash_attn_available() and is_flash_attn_version(">=", _REQUIRED_FLASH_VERSION)
_CAN_USE_FLASH_ATTN_3 = is_flash_attn_3_available()
_CAN_USE_AITER_ATTN = is_aiter_available() and is_aiter_version(">=", _REQUIRED_AITER_VERSION)
_CAN_USE_SAGE_ATTN = is_sageattention_available() and is_sageattention_version(">=", _REQUIRED_SAGE_VERSION)
_CAN_USE_FLEX_ATTN = is_torch_version(">=", _REQUIRED_FLEX_VERSION)
_CAN_USE_NPU_ATTN = is_torch_npu_available()
_CAN_USE_XLA_ATTN = is_torch_xla_available() and is_torch_xla_version(">=", _REQUIRED_XLA_VERSION)
_CAN_USE_XFORMERS_ATTN = is_xformers_available() and is_xformers_version(">=", _REQUIRED_XFORMERS_VERSION)

if _CAN_USE_FLASH_ATTN:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward, _wrapped_flash_attn_forward
else:
    flash_attn_func = None
    flash_attn_varlen_func = None
    _wrapped_flash_attn_backward = None
    _wrapped_flash_attn_forward = None

if _CAN_USE_FLASH_ATTN_3:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
else:
    flash_attn_3_func = None
    flash_attn_3_varlen_func = None

if _CAN_USE_AITER_ATTN:
    from aiter import flash_attn_func as aiter_flash_attn_func
else:
    aiter_flash_attn_func = None

if _CAN_USE_SAGE_ATTN:
    from sageattention import (
        sageattn,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp16_triton,
        sageattn_varlen,
    )
else:
    sageattn = None
    sageattn_qk_int8_pv_fp16_cuda = None
    sageattn_qk_int8_pv_fp16_triton = None
    sageattn_qk_int8_pv_fp8_cuda = None
    sageattn_qk_int8_pv_fp8_cuda_sm90 = None
    sageattn_varlen = None

if _CAN_USE_FLEX_ATTN:
    # We cannot import the flex_attention function...
    # pytorch documentation) that the user may com...
    # compiled function.
    import torch.nn.attention.flex_attention as flex_attention

if _CAN_USE_NPU_ATTN:
    from torch_npu import npu_fusion_attention
else:
    npu_fusion_attention = None

if _CAN_USE_XLA_ATTN:
    from torch_xla.experimental.custom_kernel import flash_attention as xla_flash_attention
else:
    xla_flash_attention = None

if _CAN_USE_XFORMERS_ATTN:
    import xformers.ops as xops
else:
    xops = None

# Version guard for PyTorch compatibility - custom_op was added in PyTorch 2.4
if torch.__version__ >= "2.4.0":
    _custom_op = torch.library.custom_op
    _register_fake = torch.library.register_fake
else:

    def custom_op_no_op(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func

        return wrap if fn is None else fn

    def register_fake_no_op(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        return wrap if fn is None else fn

    _custom_op = custom_op_no_op
    _register_fake = register_fake_no_op

logger = get_logger(__name__)  # pylint: disable=invalid-name

# TODO(aryan): Add support for the following:
# - Sage Attention++
# - block sparse, radial and other attention methods
# - CP with sage attention, flex, xformers, other missing backends
# - Add support for normal and CP training with backends that don't support it yet

class AttentionBackendName(str, Enum):
    # EAGER = "eager"

    # `flash-attn`
    FLASH = "flash"
    FLASH_HUB = "flash_hub"
    FLASH_VARLEN = "flash_varlen"
    FLASH_VARLEN_HUB = "flash_varlen_hub"
    _FLASH_3 = "_flash_3"
    _FLASH_VARLEN_3 = "_flash_varlen_3"
    _FLASH_3_HUB = "_flash_3_hub"
    _FLASH_3_VARLEN_HUB = "_flash_3_varlen_hub"

    # `aiter`
    AITER = "aiter"

    # PyTorch native
    FLEX = "flex"
    NATIVE = "native"
    _NATIVE_CUDNN = "_native_cudnn"
    _NATIVE_EFFICIENT = "_native_efficient"
    _NATIVE_FLASH = "_native_flash"
    _NATIVE_MATH = "_native_math"
    _NATIVE_NPU = "_native_npu"
    _NATIVE_XLA = "_native_xla"

    # `sageattention`
    SAGE = "sage"
    SAGE_HUB = "sage_hub"
    SAGE_VARLEN = "sage_varlen"
    _SAGE_QK_INT8_PV_FP8_CUDA = "_sage_qk_int8_pv_fp8_cuda"
    _SAGE_QK_INT8_PV_FP8_CUDA_SM90 = "_sage_qk_int8_pv_fp8_cuda_sm90"
    _SAGE_QK_INT8_PV_FP16_CUDA = "_sage_qk_int8_pv_fp16_cuda"
    _SAGE_QK_INT8_PV_FP16_TRITON = "_sage_qk_int8_pv_fp16_triton"
    # TODO: let's not add support for Sparge Attention now because it requires tuning per model
    # We can look into supporting something "autotune"-ing in the future
    # SPARGE = "sparge"

    # `xformers`
    XFORMERS = "xformers"

class _AttentionBackendRegistry:
    _backends = {}
    _constraints = {}
    _supported_arg_names = {}
    _supports_context_parallel = set()
    _active_backend = AttentionBackendName(DIFFUSERS_ATTN_BACKEND)
    _checks_enabled = DIFFUSERS_ATTN_CHECKS

    @classmethod
    def register(
        cls,
        backend: AttentionBackendName,
        constraints: Optional[List[Callable]] = None,
        supports_context_parallel: bool = False,
    ):
        logger.debug(f"Registering attention backend: {backend} with constraints: {constraints}")

        def decorator(func):
            cls._backends[backend] = func
            cls._constraints[backend] = constraints or []
            cls._supported_arg_names[backend] = set(inspect.signature(func).parameters.keys())
            if supports_context_parallel:
                cls._supports_context_parallel.add(backend.value)

            return func

        return decorator

    @classmethod
    def get_active_backend(cls):
        return cls._active_backend, cls._backends[cls._active_backend]

    @classmethod
    def set_active_backend(cls, backend: str):
        cls._active_backend = backend

    @classmethod
    def list_backends(cls):
        return list(cls._backends.keys())

    @classmethod
    def _is_context_parallel_available(
        cls,
        backend: AttentionBackendName,
    ) -> bool:
        supports_context_parallel = backend.value in cls._supports_context_parallel
        return supports_context_parallel

@dataclass
class _HubKernelConfig:
    class AttentionBackendName(str, Enum):
    # EAGER = "eager"

    # `flash-attn`
    FLASH = "flash"
    FLASH_HUB = "flash_hub"
    FLASH_VARLEN = "flash_varlen"
    FLASH_VARLEN_HUB = "flash_varlen_hub"
    _FLASH_3 = "_flash_3"
    _FLASH_VARLEN_3 = "_flash_varlen_3"
    _FLASH_3_HUB = "_flash_3_hub"
    _FLASH_3_VARLEN_HUB = "_flash_3_varlen_hub"

    # `aiter`
    AITER = "aiter"

    # PyTorch native
    FLEX = "flex"
    NATIVE = "native"
    _NATIVE_CUDNN = "_native_cudnn"
    _NATIVE_EFFICIENT = "_native_efficient"
    _NATIVE_FLASH = "_native_flash"
    _NATIVE_MATH = "_native_math"
    _NATIVE_NPU = "_native_npu"
    _NATIVE_XLA = "_native_xla"

    # `sageattention`
    SAGE = "sage"
    SAGE_HUB = "sage_hub"
    SAGE_VARLEN = "sage_varlen"
    _SAGE_QK_INT8_PV_FP8_CUDA = "_sage_qk_int8_pv_fp8_cuda"
    _SAGE_QK_INT8_PV_FP8_CUDA_SM90 = "_sage_qk_int8_pv_fp8_cuda_sm90"
    _SAGE_QK_INT8_PV_FP16_CUDA = "_sage_qk_int8_pv_fp16_cuda"
    _SAGE_QK_INT8_PV_FP16_TRITON = "_sage_qk_int8_pv_fp16_triton"
    # TODO: let's not add support for Sparge Attention now because it requires tuning per model
    # We can look into supporting something "autotune"-ing in the future
    # SPARGE = "sparge"

    # `xformers`
    XFORMERS = "xformers"

class _AttentionBackendRegistry:
    _backends = {}
    _constraints = {}
    _supported_arg_names = {}
    _supports_context_parallel = set()
    _active_backend = AttentionBackendName(DIFFUSERS_ATTN_BACKEND)
    _checks_enabled = DIFFUSERS_ATTN_CHECKS

    @classmethod
    def register(
        cls,
        backend: AttentionBackendName,
        constraints: Optional[List[Callable]] = None,
        supports_context_parallel: bool = False,
    ):
        logger.debug(f"Registering attention backend: {backend} with constraints: {constraints}")

        def decorator(func):
            cls._backends[backend] = func
            cls._constraints[backend] = constraints or []
            cls._supported_arg_names[backend] = set(inspect.signature(func).parameters.keys())
            if supports_context_parallel:
                cls._supports_context_parallel.add(backend.value)

            return func

        return decorator

    @classmethod
    def get_active_backend(cls):
        return cls._active_backend, cls._backends[cls._active_backend]

    @classmethod
    def set_active_backend(cls, backend: str):
        cls._active_backend = backend

    @classmethod
    def list_backends(cls):
        return list(cls._backends.keys())

    @classmethod
    def _is_context_parallel_available(
        cls,
        backend: AttentionBackendName,
    ) -> bool:
        supports_context_parallel = backend.value in cls._supports_context_parallel
        return supports_context_parallel

@dataclass
class _HubKernelConfig:


    repo_id: str
    function_attr: str
    revision: Optional[str] = None
    kernel_fn: Optional[Callable] = None

# Registry for hub-based attention kernels
_HUB_KERNELS_REGISTRY: Dict["AttentionBackendName", _HubKernelConfig] = {
    # TODO: temporary revision for now. Remove when merged upstream into `main`.
    AttentionBackendName._FLASH_3_HUB: _HubKernelConfig(
        repo_id="kernels-community/flash-attn3", function_attr="flash_attn_func", revision="fake-ops-return-probs"
    ),
    AttentionBackendName._FLASH_3_VARLEN_HUB: _HubKernelConfig(
        repo_id="kernels-community/flash-attn3",
        function_attr="flash_attn_varlen_func",
        # revision="fake-ops-return-probs",
    ),
    AttentionBackendName.FLASH_HUB: _HubKernelConfig(
        repo_id="kernels-community/flash-attn2", function_attr="flash_attn_func", revision=None
    ),
    AttentionBackendName.FLASH_VARLEN_HUB: _HubKernelConfig(
        repo_id="kernels-community/flash-attn2", function_attr="flash_attn_varlen_func", revision=None
    ),
    AttentionBackendName.SAGE_HUB: _HubKernelConfig(
        repo_id="kernels-community/sage_attention", function_attr="sageattn", revision=None
    ),
}

@contextlib.contextmanager
def attention_backend(backend: Union[str, AttentionBackendName] = AttentionBackendName.NATIVE):

    if backend not in _AttentionBackendRegistry._backends:
        raise ValueError(f"Backend {backend} is not registered.")

    backend = AttentionBackendName(backend)
    _check_attention_backend_requirements(backend)
    _maybe_download_kernel_for_backend(backend)

    old_backend = _AttentionBackendRegistry._active_backend
    _AttentionBackendRegistry.set_active_backend(backend)

    try:
        yield
    finally:
        _AttentionBackendRegistry.set_active_backend(old_backend)

def dispatch_attention_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    *,
    backend: Optional[AttentionBackendName] = None,
    parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    attention_kwargs = attention_kwargs or {}

    if backend is None:
        # If no backend is specified, we either us...
        # variable), or we use a custom backend ba...
        backend_name, backend_fn = _AttentionBackendRegistry.get_active_backend()
    else:
        backend_name = AttentionBackendName(backend)
        backend_fn = _AttentionBackendRegistry._backends.get(backend_name)

    kwargs = {
        "query": query,
        "key": key,
        "value": value,
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "scale": scale,
        **attention_kwargs,
        "_parallel_config": parallel_config,
    }
    if is_torch_version(">=", "2.5.0"):
        kwargs["enable_gqa"] = enable_gqa

    if _AttentionBackendRegistry._checks_enabled:
        removed_kwargs = set(kwargs) - set(_AttentionBackendRegistry._supported_arg_names[backend_name])
        if removed_kwargs:
            logger.warning(f"Removing unsupported arguments for attention backend {backend_name}: {removed_kwargs}.")
        for check in _AttentionBackendRegistry._constraints.get(backend_name):
            check(**kwargs)

    kwargs = {k: v for k, v in kwargs.items() if k in _AttentionBackendRegistry._supported_arg_names[backend_name]}

    return backend_fn(**kwargs)

# ===== Checks =====
# A list of very simple functions to catch common errors quickly when debugging.

def _check_attn_mask_or_causal(attn_mask: Optional[torch.Tensor], is_causal: bool, **kwargs) -> None:
    if attn_mask is not None and is_causal:
        raise ValueError("`is_causal` cannot be True when `attn_mask` is not None.")

def _check_device(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    if query.device != key.device or query.device != value.device:
        raise ValueError("Query, key, and value must be on the same device.")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Query, key, and value must have the same dtype.")

def _check_device_cuda(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    _check_device(query, key, value)
    if query.device.type != "cuda":
        raise ValueError("Query, key, and value must be on a CUDA device.")

def _check_device_cuda_atleast_smXY(major: int, minor: int) -> Callable:
    def check_device_cuda(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
        _check_device_cuda(query, key, value)
        if torch.cuda.get_device_capability(query.device) < (major, minor):
            raise ValueError(
                f"Query, key, and value must be on a CUDA device with compute capability >= {major}.{minor}."
            )

    return check_device_cuda

def _check_qkv_dtype_match(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    if query.dtype != key.dtype:
        raise ValueError("Query and key must have the same dtype.")
    if query.dtype != value.dtype:
        raise ValueError("Query and value must have the same dtype.")

def _check_qkv_dtype_bf16_or_fp16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    _check_qkv_dtype_match(query, key, value)
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Query, key, and value must be either bfloat16 or float16.")

def _check_shape(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> None:
    # Expected shapes:
    # query: (batch_size, seq_len_q, num_heads, head_dim)
    # key:   (batch_size, seq_len_kv, num_heads, head_dim)
    # value: (batch_size, seq_len_kv, num_heads, head_dim)
    # attn_mask: (seq_len_q, seq_len_kv) or (batch_size, seq_len_q, seq_len_kv)
    #            or (batch_size, num_heads, seq_len_q, seq_len_kv)
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Query and key must have the same head dimension.")
    if key.shape[-3] != value.shape[-3]:
        raise ValueError("Key and value must have the same sequence length.")
    if attn_mask is not None and attn_mask.shape[-1] != key.shape[-3]:
        raise ValueError("Attention mask must match the key's sequence length.")

# ===== Helper functions =====

def _check_attention_backend_requirements(backend: AttentionBackendName) -> None:
    if backend in [AttentionBackendName.FLASH, AttentionBackendName.FLASH_VARLEN]:
        if not _CAN_USE_FLASH_ATTN:
            raise RuntimeError(
                f"Flash Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `flash-attn>={_REQUIRED_FLASH_VERSION}`."
            )

    elif backend in [AttentionBackendName._FLASH_3, AttentionBackendName._FLASH_VARLEN_3]:
        if not _CAN_USE_FLASH_ATTN_3:
            raise RuntimeError(
                f"Flash Attention 3 backend '{backend.value}' is not usable because of missing package or the version is too old. Please build FA3 beta release from source."
            )

    elif backend in [
        AttentionBackendName.FLASH_HUB,
        AttentionBackendName.FLASH_VARLEN_HUB,
        AttentionBackendName._FLASH_3_HUB,
        AttentionBackendName._FLASH_3_VARLEN_HUB,
        AttentionBackendName.SAGE_HUB,
    ]:
        if not is_kernels_available():
            raise RuntimeError(
                f"Backend '{backend.value}' is not usable because the `kernels` package isn't available. Please install it with `pip install kernels`."
            )

    elif backend == AttentionBackendName.AITER:
        if not _CAN_USE_AITER_ATTN:
            raise RuntimeError(
                f"Aiter Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `aiter>={_REQUIRED_AITER_VERSION}`."
            )

    elif backend in [
        AttentionBackendName.SAGE,
        AttentionBackendName.SAGE_VARLEN,
        AttentionBackendName._SAGE_QK_INT8_PV_FP8_CUDA,
        AttentionBackendName._SAGE_QK_INT8_PV_FP8_CUDA_SM90,
        AttentionBackendName._SAGE_QK_INT8_PV_FP16_CUDA,
        AttentionBackendName._SAGE_QK_INT8_PV_FP16_TRITON,
    ]:
        if not _CAN_USE_SAGE_ATTN:
            raise RuntimeError(
                f"Sage Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `sageattention>={_REQUIRED_SAGE_VERSION}`."
            )

    elif backend == AttentionBackendName.FLEX:
        if not _CAN_USE_FLEX_ATTN:
            raise RuntimeError(
                f"Flex Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `torch>=2.5.0`."
            )

    elif backend == AttentionBackendName._NATIVE_NPU:
        if not _CAN_USE_NPU_ATTN:
            raise RuntimeError(
                f"NPU Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `torch_npu`."
            )

    elif backend == AttentionBackendName._NATIVE_XLA:
        if not _CAN_USE_XLA_ATTN:
            raise RuntimeError(
                f"XLA Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `torch_xla>={_REQUIRED_XLA_VERSION}`."
            )

    elif backend == AttentionBackendName.XFORMERS:
        if not _CAN_USE_XFORMERS_ATTN:
            raise RuntimeError(
                f"Xformers Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `xformers>={_REQUIRED_XFORMERS_VERSION}`."
            )

@functools.lru_cache(maxsize=128)
def _prepare_for_flash_attn_or_sage_varlen_without_mask(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    device: Optional[torch.device] = None,
):
    seqlens_q = torch.full((batch_size,), seq_len_q, dtype=torch.int32, device=device)
    seqlens_k = torch.full((batch_size,), seq_len_kv, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)
    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()
    return (seqlens_q, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)

def _prepare_for_flash_attn_or_sage_varlen_with_mask(
    batch_size: int,
    seq_len_q: int,
    attn_mask: torch.Tensor,
    device: Optional[torch.device] = None,
):
    seqlens_q = torch.full((batch_size,), seq_len_q, dtype=torch.int32, device=device)
    seqlens_k = attn_mask.sum(dim=1, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)
    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()
    return (seqlens_q, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)

def _prepare_for_flash_attn_or_sage_varlen(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    attn_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> None:
    if attn_mask is None:
        return _prepare_for_flash_attn_or_sage_varlen_without_mask(batch_size, seq_len_q, seq_len_kv, device)
    return _prepare_for_flash_attn_or_sage_varlen_with_mask(batch_size, seq_len_q, attn_mask, device)

def _normalize_attn_mask(attn_mask: torch.Tensor, batch_size: int, seq_len_k: int) -> torch.Tensor:

    if attn_mask.dtype != torch.bool:
        raise ValueError(f"Attention mask must be of type bool, got {attn_mask.dtype}.")

    if attn_mask.ndim == 1:
        # [seq_len_k] -> broadcast across batch
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, seq_len_k)

    elif attn_mask.ndim == 2:
        # [batch_size, seq_len_k]. Maybe broadcast across batch
        if attn_mask.size(0) not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 2D attention mask."
            )
        attn_mask = attn_mask.expand(batch_size, seq_len_k)

    elif attn_mask.ndim == 3:
        # [batch_size, seq_len_q, seq_len_k] -> reduce over query dimension
        # We do this reduction because we know tha...
        if attn_mask.size(0) not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 3D attention mask."
            )
        attn_mask = attn_mask.any(dim=1)
        attn_mask = attn_mask.expand(batch_size, seq_len_k)

    elif attn_mask.ndim == 4:
        # [batch_size, num_heads, seq_len_q, seq_len_k] or broadcastable versions
        if attn_mask.size(0) not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 4D attention mask."
            )
        attn_mask = attn_mask.expand(batch_size, -1, -1, seq_len_k)  # [B, H, Q, K]
        attn_mask = attn_mask.any(dim=(1, 2))  # [B, K]

    else:
        raise ValueError(f"Unsupported attention mask shape: {attn_mask.shape}")

    if attn_mask.shape != (batch_size, seq_len_k):
        raise ValueError(
            f"Normalized attention mask shape mismatch: got {attn_mask.shape}, expected ({batch_size}, {seq_len_k})"
        )

    return attn_mask

def _flex_attention_causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
    return q_idx >= kv_idx

# ===== Helpers for downloading kernels =====
def _maybe_download_kernel_for_backend(backend: AttentionBackendName) -> None:
    if backend not in _HUB_KERNELS_REGISTRY:
        return
    config = _HUB_KERNELS_REGISTRY[backend]

    if config.kernel_fn is not None:
        return

    try:
        from kernels import get_kernel

        kernel_module = get_kernel(config.repo_id, revision=config.revision)
        kernel_func = getattr(kernel_module, config.function_attr)

        # Cache the downloaded kernel function in the config object
        config.kernel_fn = kernel_func

    except Exception as e:
        logger.error(f"An error occurred while fetching kernel '{config.repo_id}' from the Hub: {e}")
        raise

# ===== torch op registrations =====
# Registrations are required for fullgraph tracing compatibility
# TODO: this is only required because the beta release FA3 does not have it. There is a PR adding
# this but it was never merged: https://github.com/Dao-AILab/flash-attention/pull/1590
@_custom_op("_diffusers_flash_attn_3::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _wrapped_flash_attn_3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Hardcoded for now because pytorch does not support tuple/int type hints
    window_size = (-1, -1)
    out, lse, *_ = flash_attn_3_func(
        q=q,
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
        sm_margin=sm_margin,
    )
    lse = lse.permute(0, 2, 1)
    return out, lse

@_register_fake("_diffusers_flash_attn_3::_flash_attn_forward")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    window_size = (-1, -1)  # noqa: F841
    # A lot of the parameters here are not yet used in any way within diffusers.
    # We can safely ignore for now and keep the fake op shape propagation simple.
    batch_size, seq_len, num_heads, head_dim = q.shape
    lse_shape = (batch_size, seq_len, num_heads)
    return torch.empty_like(q), q.new_empty(lse_shape)

# ===== Helper functions to use attention backends with templated CP autograd functions =====

def _native_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    # Native attention does not return_lse
    if return_lse:
        raise ValueError("Native attention does not support return_lse=True")

    # used for backward pass
    if _save_ctx:
        ctx.save_for_backward(query, key, value)
        ctx.attn_mask = attn_mask
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa

    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    out = out.permute(0, 2, 1, 3)

    return out

def _native_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
    **kwargs,
):
    query, key, value = ctx.saved_tensors

    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    query_t, key_t, value_t = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = torch.nn.functional.scaled_dot_product_attention(
        query=query_t,
        key=key_t,
        value=value_t,
        attn_mask=ctx.attn_mask,
        dropout_p=ctx.dropout_p,
        is_causal=ctx.is_causal,
        scale=ctx.scale,
        enable_gqa=ctx.enable_gqa,
    )
    out = out.permute(0, 2, 1, 3)

    grad_out_t = grad_out.permute(0, 2, 1, 3)
    grad_query_t, grad_key_t, grad_value_t = torch.autograd.grad(
        outputs=out, inputs=[query_t, key_t, value_t], grad_outputs=grad_out_t, retain_graph=False
    )

    grad_query = grad_query_t.permute(0, 2, 1, 3)
    grad_key = grad_key_t.permute(0, 2, 1, 3)
    grad_value = grad_value_t.permute(0, 2, 1, 3)

    return grad_query, grad_key, grad_value

# https://github.com/pytorch/pytorch/blob/8904ba63...
# forward declaration:
#   aten::_scaled_dot_product_cudnn_attention(Tens...
def _cudnn_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    if enable_gqa:
        raise ValueError("`enable_gqa` is not yet supported for cuDNN attention.")

    tensors_to_save = ()

    # Contiguous is a must here! Calling cuDNN backend with aten ops produces incorrect results
    # if the input tensors are not contiguous.
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    tensors_to_save += (query, key, value)

    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_cudnn_attention(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_mask,
            compute_log_sumexp=return_lse,
            dropout_p=dropout_p,
            is_causal=is_causal,
            return_debug_mask=False,
            scale=scale,
        )
    )

    tensors_to_save += (out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset)
    if _save_ctx:
        ctx.save_for_backward(*tensors_to_save)
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.attn_mask = attn_mask
        ctx.max_q = max_q
        ctx.max_k = max_k

    out = out.transpose(1, 2).contiguous()
    if lse is not None:
        lse = lse.transpose(1, 2).contiguous()
    return (out, lse) if return_lse else out

# backward declaration:
#   aten::_scaled_dot_product_cudnn_attention_back...
def _cudnn_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
    **kwargs,
):
    query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset = ctx.saved_tensors

    grad_out = grad_out.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    # Cannot pass first 5 arguments as kwargs beca...
    grad_query, grad_key, grad_value = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp=lse,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        attn_bias=ctx.attn_mask,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=ctx.max_q,
        max_k=ctx.max_k,
        dropout_p=ctx.dropout_p,
        is_causal=ctx.is_causal,
        scale=ctx.scale,
    )
    grad_query, grad_key, grad_value = (x.transpose(1, 2).contiguous() for x in (grad_query, grad_key, grad_value))

    return grad_query, grad_key, grad_value

# https://github.com/pytorch/pytorch/blob/e33fa0ec...
# forward declaration:
#   aten::_scaled_dot_product_flash_attention(Tens...
def _native_flash_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    if enable_gqa:
        raise ValueError("`enable_gqa` is not yet supported for native flash attention.")

    tensors_to_save = ()

    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    tensors_to_save += (query, key, value)

    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_flash_attention(
            query=query,
            key=key,
            value=value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            return_debug_mask=False,
            scale=scale,
        )
    )

    tensors_to_save += (out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset)
    if _save_ctx:
        ctx.save_for_backward(*tensors_to_save)
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.max_q = max_q
        ctx.max_k = max_k

    out = out.transpose(1, 2).contiguous()
    if lse is not None:
        lse = lse.transpose(1, 2).contiguous()
    return (out, lse) if return_lse else out

# https://github.com/pytorch/pytorch/blob/e33fa0ec...
# backward declaration:
#   aten::_scaled_dot_product_flash_attention_back...
def _native_flash_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
    **kwargs,
):
    query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset = ctx.saved_tensors

    grad_out = grad_out.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    grad_query, grad_key, grad_value = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp=lse,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=ctx.max_q,
        max_k=ctx.max_k,
        dropout_p=ctx.dropout_p,
        is_causal=ctx.is_causal,
        scale=ctx.scale,
    )
    grad_query, grad_key, grad_value = (x.transpose(1, 2).contiguous() for x in (grad_query, grad_key, grad_value))

    return grad_query, grad_key, grad_value

# Adapted from: https://github.com/Dao-AILab/flash...
def _flash_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not yet supported for flash-attn 2.")
    if enable_gqa:
        raise ValueError("`enable_gqa` is not yet supported for flash-attn 2.")

    # Hardcoded for now
    window_size = (-1, -1)
    softcap = 0.0
    alibi_slopes = None
    deterministic = False
    grad_enabled = any(x.requires_grad for x in (query, key, value))

    if scale is None:
        scale = query.shape[-1] ** (-0.5)

    # flash-attn only returns LSE if dropout_p > 0. So, we need to workaround.
    if grad_enabled or (_parallel_config is not None and _parallel_config.context_parallel_config._world_size > 1):
        dropout_p = dropout_p if dropout_p > 0 else 1e-30

    with torch.set_grad_enabled(grad_enabled):
        out, lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            query,
            key,
            value,
            dropout_p,
            scale,
            is_causal,
            window_size[0],
            window_size[1],
            softcap,
            alibi_slopes,
            return_lse,
        )
        lse = lse.permute(0, 2, 1)

    if _save_ctx:
        ctx.save_for_backward(query, key, value, out, lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic

    return (out, lse) if return_lse else out

def _flash_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
    **kwargs,
):
    query, key, value, out, lse, rng_state = ctx.saved_tensors
    grad_query, grad_key, grad_value = torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)

    lse_d = _wrapped_flash_attn_backward(  # noqa: F841
        grad_out,
        query,
        key,
        value,
        out,
        lse,
        grad_query,
        grad_key,
        grad_value,
        ctx.dropout_p,
        ctx.scale,
        ctx.is_causal,
        ctx.window_size[0],
        ctx.window_size[1],
        ctx.softcap,
        ctx.alibi_slopes,
        ctx.deterministic,
        rng_state,
    )

    # Head dimension may have been padded
    grad_query = grad_query[..., : grad_out.shape[-1]]
    grad_key = grad_key[..., : grad_out.shape[-1]]
    grad_value = grad_value[..., : grad_out.shape[-1]]

    return grad_query, grad_key, grad_value

def _sage_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not yet supported for Sage attention.")
    if dropout_p > 0.0:
        raise ValueError("`dropout_p` is not yet supported for Sage attention.")
    if enable_gqa:
        raise ValueError("`enable_gqa` is not yet supported for Sage attention.")

    out = sageattn(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        sm_scale=scale,
        return_lse=return_lse,
    )
    lse = None
    if return_lse:
        out, lse, *_ = out
        lse = lse.permute(0, 2, 1)

    return (out, lse) if return_lse else out

def _sage_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
):
    raise NotImplementedError("Backward pass is not implemented for Sage attention.")

def _npu_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    if return_lse:
        raise ValueError("NPU attention backend does not support setting `return_lse=True`.")

    out = npu_fusion_attention(
        query,
        key,
        value,
        query.size(2),  # num_heads
        input_layout="BSND",
        pse=None,
        scale=1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
        pre_tockens=65536,
        next_tockens=65536,
        keep_prob=1.0 - dropout_p,
        sync=False,
        inner_precise=0,
    )[0]

    return out

# Not implemented yet.
def _npu_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
    **kwargs,
):
    raise NotImplementedError("Backward pass is not implemented for Npu Fusion Attention.")

# ===== Context parallel =====

# Reference:
# - https://github.com/pytorch/pytorch/blob/f58a68...
# - https://github.com/pytorch/pytorch/blob/f58a68...
# For fullgraph=True tracing compatibility (since FakeTensor does not have a `wait` method):
def _wait_tensor(tensor):
    if isinstance(tensor, funcol.AsyncCollectiveTensor):
        tensor = tensor.wait()
    return tensor

def _all_to_all_single(x: torch.Tensor, group) -> torch.Tensor:
    shape = x.shape
    # HACK: We need to flatten because despite making tensors contiguous, torch single-file-ization
    # to benchmark triton codegen fails somewhere:
    # buf25 = torch.ops._c10d_functional.all_to_all_single.default(buf24, [1, 1], [1, 1], '3')
    # ValueError: Tensors must be contiguous
    x = x.flatten()
    x = funcol.all_to_all_single(x, None, None, group)
    x = x.reshape(shape)
    x = _wait_tensor(x)
    return x

def _all_to_all_dim_exchange(x: torch.Tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None) -> torch.Tensor:

    Unified Sequence Parallelism attention combining Ulysses and ring attention. See: https://arxiv.org/abs/2405.07719

    Convert a 2D attention mask to an additive mask, optionally reshaping to 4D for SDPA.

    This helper is used by both native SDPA and xformers backends to handle both boolean and additive masks.
                   - Boolean: True means attend, False means mask out
                   - Additive: 0.0 means attend, -inf means mask out
        target_dtype: The dtype to convert the mask to (usually query.dtype)
        reshape_4d: If True, reshape from [batch_size, seq_len_k] to [batch_size, 1, 1, seq_len_k] for broadcasting
)
def _native_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("Native attention backend does not support setting `return_lse=True`.")

    # Reshape 2D mask to 4D for SDPA
    # SDPA accepts both boolean masks (torch.bool) and additive masks (float)
    if (
        attn_mask is not None
        and attn_mask.ndim == 2
        and attn_mask.shape[0] == query.shape[0]
        and attn_mask.shape[1] == key.shape[1]
    ):
        # Just reshape [batch_size, seq_len_k] -> [batch_size, 1, 1, seq_len_k]
        # SDPA handles both boolean and additive masks correctly
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

    if _parallel_config is None:
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        out = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        out = out.permute(0, 2, 1, 3)
    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            forward_op=_native_attention_forward_op,
            backward_op=_native_attention_backward_op,
            _parallel_config=_parallel_config,
        )

    return out

@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_CUDNN,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
)
def _native_cudnn_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    lse = None
    if _parallel_config is None and not return_lse:
        query, key, value = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        out = out.permute(0, 2, 1, 3)
    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            forward_op=_cudnn_attention_forward_op,
            backward_op=_cudnn_attention_backward_op,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse = out

    return (out, lse) if return_lse else out

@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_EFFICIENT,
    constraints=[_check_device, _check_shape],
)
def _native_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("Native efficient attention backend does not support setting `return_lse=True`.")
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    out = out.permute(0, 2, 1, 3)
    return out

@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_FLASH,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
)
def _native_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not supported for aiter attention")

    lse = None
    if _parallel_config is None and not return_lse:
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,  # not supported
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        out = out.permute(0, 2, 1, 3)
    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            None,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            forward_op=_native_flash_attention_forward_op,
            backward_op=_native_flash_attention_backward_op,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse = out

    return (out, lse) if return_lse else out

@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_MATH,
    constraints=[_check_device, _check_shape],
)
def _native_math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("Native math attention backend does not support setting `return_lse=True`.")
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    out = out.permute(0, 2, 1, 3)
    return out

@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_NPU,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
)
def _native_npu_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not supported for NPU attention")
    if return_lse:
        raise ValueError("NPU attention backend does not support setting `return_lse=True`.")
    if _parallel_config is None:
        out = npu_fusion_attention(
            query,
            key,
            value,
            query.size(2),  # num_heads
            input_layout="BSND",
            pse=None,
            scale=1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
            pre_tockens=65536,
            next_tockens=65536,
            keep_prob=1.0 - dropout_p,
            sync=False,
            inner_precise=0,
        )[0]
    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            None,
            dropout_p,
            None,
            scale,
            None,
            return_lse,
            forward_op=_npu_attention_forward_op,
            backward_op=_npu_attention_backward_op,
            _parallel_config=_parallel_config,
        )
    return out

# Reference: https://github.com/pytorch/xla/blob/0...
@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_XLA,
    constraints=[_check_device, _check_shape],
)
def _native_xla_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not supported for XLA attention")
    if return_lse:
        raise ValueError("XLA attention backend does not support setting `return_lse=True`.")
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    query = query / math.sqrt(query.shape[-1])
    out = xla_flash_attention(
        q=query,
        k=key,
        v=value,
        causal=is_causal,
    )
    out = out.permute(0, 2, 1, 3)
    return out

@_AttentionBackendRegistry.register(
    AttentionBackendName.SAGE,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
)
def _sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not supported for sage attention")
    lse = None
    if _parallel_config is None:
        out = sageattn(
            q=query,
            k=key,
            v=value,
            tensor_layout="NHD",
            is_causal=is_causal,
            sm_scale=scale,
            return_lse=return_lse,
        )
        if return_lse:
            out, lse, *_ = out
    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            None,
            0.0,
            is_causal,
            scale,
            False,
            return_lse,
            forward_op=_sage_attention_forward_op,
            backward_op=_sage_attention_backward_op,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse = out

    return (out, lse) if return_lse else out

@_AttentionBackendRegistry.register(
    AttentionBackendName.SAGE_HUB,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=False,
)
def _sage_attention_hub(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not supported for sage attention")
    lse = None
    func = _HUB_KERNELS_REGISTRY[AttentionBackendName.SAGE_HUB].kernel_fn
    if _parallel_config is None:
        out = func(
            q=query,
            k=key,
            v=value,
            tensor_layout="NHD",
            is_causal=is_causal,
            sm_scale=scale,
            return_lse=return_lse,
        )
        if return_lse:
            out, lse, *_ = out

    return (out, lse) if return_lse else out

@_AttentionBackendRegistry.register(
    AttentionBackendName.SAGE_VARLEN,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _sage_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("Sage varlen backend does not support setting `return_lse=True`.")

    batch_size, seq_len_q, _, _ = query.shape
    _, seq_len_kv, _, _ = key.shape

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

    (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
        _prepare_for_flash_attn_or_sage_varlen(
            batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
        )
    )

    key_valid, value_valid = [], []
    for b in range(batch_size):
        valid_len = seqlens_k[b]
        key_valid.append(key[b, :valid_len])
        value_valid.append(value[b, :valid_len])

    query_packed = query.flatten(0, 1)
    key_packed = torch.cat(key_valid, dim=0)
    value_packed = torch.cat(value_valid, dim=0)

    out = sageattn_varlen(
        q=query_packed,
        k=key_packed,
        v=value_packed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        is_causal=is_causal,
        sm_scale=scale,
    )
    out = out.unflatten(0, (batch_size, -1))

    return out

@_AttentionBackendRegistry.register(
    AttentionBackendName._SAGE_QK_INT8_PV_FP8_CUDA,
    constraints=[_check_device_cuda_atleast_smXY(9, 0), _check_shape],
)
def _sage_qk_int8_pv_fp8_cuda_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not supported for sage attention")
    return sageattn_qk_int8_pv_fp8_cuda(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        sm_scale=scale,
        return_lse=return_lse,
    )

@_AttentionBackendRegistry.register(
    AttentionBackendName._SAGE_QK_INT8_PV_FP8_CUDA_SM90,
    constraints=[_check_device_cuda_atleast_smXY(9, 0), _check_shape],
)
def _sage_qk_int8_pv_fp8_cuda_sm90_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not supported for sage attention")
    return sageattn_qk_int8_pv_fp8_cuda_sm90(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        sm_scale=scale,
        return_lse=return_lse,
    )

@_AttentionBackendRegistry.register(
    AttentionBackendName._SAGE_QK_INT8_PV_FP16_CUDA,
    constraints=[_check_device_cuda_atleast_smXY(8, 0), _check_shape],
)
def _sage_qk_int8_pv_fp16_cuda_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not supported for sage attention")
    return sageattn_qk_int8_pv_fp16_cuda(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        sm_scale=scale,
        return_lse=return_lse,
    )

@_AttentionBackendRegistry.register(
    AttentionBackendName._SAGE_QK_INT8_PV_FP16_TRITON,
    constraints=[_check_device_cuda_atleast_smXY(8, 0), _check_shape],
)
def _sage_qk_int8_pv_fp16_triton_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not supported for sage attention")
    return sageattn_qk_int8_pv_fp16_triton(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        sm_scale=scale,
        return_lse=return_lse,
    )

@_AttentionBackendRegistry.register(
    AttentionBackendName.XFORMERS,
    constraints=[_check_attn_mask_or_causal, _check_device, _check_shape],
)
def _xformers_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("xformers attention backend does not support setting `return_lse=True`.")

    batch_size, seq_len_q, num_heads_q, _ = query.shape
    _, seq_len_kv, num_heads_kv, _ = key.shape

    if is_causal:
        attn_mask = xops.LowerTriangularMask()
    elif attn_mask is not None:
        if attn_mask.ndim == 2:
            # Convert 2D mask to 4D for xformers
            # Mask can be boolean (True=attend, False=mask) or additive (0.0=attend, -inf=mask)
            # xformers requires 4D additive masks [batch, heads, seq_q, seq_k]
            # Need memory alignment - create larger tensor and slice for alignment
            original_seq_len = attn_mask.size(1)
            aligned_seq_len = ((original_seq_len + 7) // 8) * 8  # Round up to multiple of 8

            # Create aligned 4D tensor and slice to ensure proper memory layout
            aligned_mask = torch.zeros(
                (batch_size, num_heads_q, seq_len_q, aligned_seq_len),
                dtype=query.dtype,
                device=query.device,
            )
            # Convert to 4D additive mask (handles both boolean and additive inputs)
            mask_additive = _prepare_additive_attn_mask(
                attn_mask, target_dtype=query.dtype
            )  # [batch, 1, 1, seq_len_k]
            # Broadcast to [batch, heads, seq_q, seq_len_k]
            aligned_mask[:, :, :, :original_seq_len] = mask_additive
            # Mask out the padding (already -inf from zeros -> where with default)
            aligned_mask[:, :, :, original_seq_len:] = float("-inf")

            # Slice to actual size with proper alignment
            attn_mask = aligned_mask[:, :, :, :seq_len_kv]
        elif attn_mask.ndim != 4:
            raise ValueError("Only 2D and 4D attention masks are supported for xformers attention.")
        elif attn_mask.ndim == 4:
            attn_mask = attn_mask.expand(batch_size, num_heads_q, seq_len_q, seq_len_kv).type_as(query)

    if enable_gqa:
        if num_heads_q % num_heads_kv != 0:
            raise ValueError("Number of heads in query must be divisible by number of heads in key/value.")
        num_heads_per_group = num_heads_q // num_heads_kv
        query = query.unflatten(2, (num_heads_kv, -1))
        key = key.unflatten(2, (num_heads_kv, -1)).expand(-1, -1, -1, num_heads_per_group, -1)
        value = value.unflatten(2, (num_heads_kv, -1)).expand(-1, -1, -1, num_heads_per_group, -1)

    out = xops.memory_efficient_attention(query, key, value, attn_mask, dropout_p, scale)

    if enable_gqa:
        out = out.flatten(2, 3)

    return out
