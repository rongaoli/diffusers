#!/usr/bin/env python
# coding=utf-8


import copy
import dataclasses
import importlib.metadata
import inspect
import json
import os
import warnings
from dataclasses import dataclass, is_dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

from packaging import version

from ..utils import is_torch_available, is_torchao_available, is_torchao_version, logging

if is_torch_available():
    import torch

logger = logging.get_logger(__name__)

class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GGUF = "gguf"
    TORCHAO = "torchao"
    QUANTO = "quanto"
    MODELOPT = "modelopt"

if is_torchao_available():
    from torchao.quantization.quant_primitives import MappingType

    class TorchAoJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, MappingType):
                return obj.name
            return super().default(obj)

@dataclass
class QuantizationConfigMixin:
    class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GGUF = "gguf"
    TORCHAO = "torchao"
    QUANTO = "quanto"
    MODELOPT = "modelopt"

if is_torchao_available():
    from torchao.quantization.quant_primitives import MappingType

    class TorchAoJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, MappingType):
                return obj.name
            return super().default(obj)

@dataclass
class QuantizationConfigMixin:
    
    class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GGUF = "gguf"
    TORCHAO = "torchao"
    QUANTO = "quanto"
    MODELOPT = "modelopt"

if is_torchao_available():
    from torchao.quantization.quant_primitives import MappingType

    class TorchAoJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, MappingType):
                return obj.name
            return super().default(obj)

@dataclass
class QuantizationConfigMixin:
    class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GGUF = "gguf"
    TORCHAO = "torchao"
    QUANTO = "quanto"
    MODELOPT = "modelopt"

if is_torchao_available():
    from torchao.quantization.quant_primitives import MappingType

    class TorchAoJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, MappingType):
                return obj.name
            return super().default(obj)

@dataclass
class QuantizationConfigMixin:


    quant_method: QuantizationMethod
    _exclude_attributes_at_init = []

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):


    _exclude_attributes_at_init = ["_load_in_4bit", "_load_in_8bit", "quant_method"]

    def __init__(
        self,
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.BITS_AND_BYTES

        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")

        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        if bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = torch.float32
        elif isinstance(bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        elif isinstance(bnb_4bit_compute_dtype, torch.dtype):
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        else:
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        if bnb_4bit_quant_storage is None:
            self.bnb_4bit_quant_storage = torch.uint8
        elif isinstance(bnb_4bit_quant_storage, str):
            if bnb_4bit_quant_storage not in [
                "float16",
                "float32",
                "int8",
                "uint8",
                "float64",
                "bfloat16",
            ]:
                raise ValueError(
                    "`bnb_4bit_quant_storage` must be a valid string (one of 'float16', 'float32', 'int8', 'uint8', 'float64', 'bfloat16') "
                )
            self.bnb_4bit_quant_storage = getattr(torch, bnb_4bit_quant_storage)
        elif isinstance(bnb_4bit_quant_storage, torch.dtype):
            self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        else:
            raise ValueError("bnb_4bit_quant_storage must be a string or a torch.dtype")

        if kwargs and not all(k in self._exclude_attributes_at_init for k in kwargs):
            logger.warning(f"Unused kwargs: {list(kwargs.keys())}. These kwargs are not used in {self.__class__}.")

        self.post_init()

    @property
    def load_in_4bit(self):
        return self._load_in_4bit

    @load_in_4bit.setter
    def load_in_4bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if self.load_in_8bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_4bit = value

    @property
    def load_in_8bit(self):
        return self._load_in_8bit

    @load_in_8bit.setter
    def load_in_8bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if self.load_in_4bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_8bit = value

    def post_init(self):

        if not isinstance(self.load_in_4bit, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if not isinstance(self.load_in_8bit, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if not isinstance(self.llm_int8_threshold, float):
            raise TypeError("llm_int8_threshold must be a float")

        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise TypeError("llm_int8_skip_modules must be a list of strings")
        if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
            raise TypeError("llm_int8_enable_fp32_cpu_offload must be a boolean")

        if not isinstance(self.llm_int8_has_fp16_weight, bool):
            raise TypeError("llm_int8_has_fp16_weight must be a boolean")

        if self.bnb_4bit_compute_dtype is not None and not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise TypeError("bnb_4bit_compute_dtype must be torch.dtype")

        if not isinstance(self.bnb_4bit_quant_type, str):
            raise TypeError("bnb_4bit_quant_type must be a string")

        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise TypeError("bnb_4bit_use_double_quant must be a boolean")

        if self.load_in_4bit and not version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse(
            "0.39.0"
        ):
            raise ValueError(
                "4 bit quantization requires bitsandbytes>=0.39.0 - please upgrade your bitsandbytes version"
            )

    def is_quantizable(self):

        return self.load_in_8bit or self.load_in_4bit

    def quantization_method(self):

        if self.load_in_8bit:
            return "llm_int8"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "fp4":
            return "fp4"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "nf4":
            return "nf4"
        else:
            return None

    def to_dict(self) -> Dict[str, Any]:

        output = copy.deepcopy(self.__dict__)
        output["bnb_4bit_compute_dtype"] = str(output["bnb_4bit_compute_dtype"]).split(".")[1]
        output["bnb_4bit_quant_storage"] = str(output["bnb_4bit_quant_storage"]).split(".")[1]
        output["load_in_4bit"] = self.load_in_4bit
        output["load_in_8bit"] = self.load_in_8bit

        return output

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> Dict[str, Any]:


    def __init__(self, compute_dtype: Optional["torch.dtype"] = None):
        self.quant_method = QuantizationMethod.GGUF
        self.compute_dtype = compute_dtype
        self.pre_quantized = True

        # TODO: (Dhruv) Add this as an init argument when we can support loading unquantized checkpoints.
        self.modules_to_not_convert = None

        if self.compute_dtype is None:
            self.compute_dtype = torch.float32

@dataclass
class TorchAoConfig(QuantizationConfigMixin):
    
    class TorchAoConfig(QuantizationConfigMixin):


    def __init__(
        self,
        quant_type: Union[str, "AOBaseConfig"],  # noqa: F821
        modules_to_not_convert: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.quant_method = QuantizationMethod.TORCHAO
        self.quant_type = quant_type
        self.modules_to_not_convert = modules_to_not_convert

        # When we load from serialized config, "quant_type_kwargs" will be the key
        if "quant_type_kwargs" in kwargs:
            self.quant_type_kwargs = kwargs["quant_type_kwargs"]
        else:
            self.quant_type_kwargs = kwargs

        self.post_init()

    def post_init(self):
        if not isinstance(self.quant_type, str):
            if is_torchao_version("<=", "0.9.0"):
                raise ValueError(
                    f"torchao <= 0.9.0 only supports string quant_type, got {type(self.quant_type).__name__}. "
                    f"Upgrade to torchao > 0.9.0 to use AOBaseConfig."
                )

            from torchao.quantization.quant_api import AOBaseConfig

            if not isinstance(self.quant_type, AOBaseConfig):
                raise TypeError(f"quant_type must be a AOBaseConfig instance, got {type(self.quant_type).__name__}")

        elif isinstance(self.quant_type, str):
            TORCHAO_QUANT_TYPE_METHODS = self._get_torchao_quant_type_to_method()

            if self.quant_type not in TORCHAO_QUANT_TYPE_METHODS.keys():
                is_floatx_quant_type = self.quant_type.startswith("fp")
                is_float_quant_type = self.quant_type.startswith("float") or is_floatx_quant_type
                if is_float_quant_type and not self._is_xpu_or_cuda_capability_atleast_8_9():
                    raise ValueError(
                        f"Requested quantization type: {self.quant_type} is not supported on GPUs with CUDA capability <= 8.9. You "
                        f"can check the CUDA capability of your GPU using `torch.cuda.get_device_capability()`."
                    )
                elif is_floatx_quant_type and not is_torchao_version("<=", "0.14.1"):
                    raise ValueError(
                        f"Requested quantization type: {self.quant_type} is only supported in torchao <= 0.14.1. "
                        f"Please downgrade to torchao <= 0.14.1 to use this quantization type."
                    )

                raise ValueError(
                    f"Requested quantization type: {self.quant_type} is not supported or is an incorrect `quant_type` name. If you think the "
                    f"provided quantization type should be supported, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            method = TORCHAO_QUANT_TYPE_METHODS[self.quant_type]
            signature = inspect.signature(method)
            all_kwargs = {
                param.name
                for param in signature.parameters.values()
                if param.kind in [inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]
            }
            unsupported_kwargs = list(self.quant_type_kwargs.keys() - all_kwargs)

            if len(unsupported_kwargs) > 0:
                raise ValueError(
                    f'The quantization method "{self.quant_type}" does not support the following keyword arguments: '
                    f"{unsupported_kwargs}. The following keywords arguments are supported: {all_kwargs}."
                )

    def to_dict(self):

        d = super().to_dict()

        if isinstance(self.quant_type, str):
            # Handle layout serialization if present
            if "quant_type_kwargs" in d and "layout" in d["quant_type_kwargs"]:
                if is_dataclass(d["quant_type_kwargs"]["layout"]):
                    d["quant_type_kwargs"]["layout"] = [
                        d["quant_type_kwargs"]["layout"].__class__.__name__,
                        dataclasses.asdict(d["quant_type_kwargs"]["layout"]),
                    ]
                if isinstance(d["quant_type_kwargs"]["layout"], list):
                    assert len(d["quant_type_kwargs"]["layout"]) == 2, "layout saves layout name and layout kwargs"
                    assert isinstance(d["quant_type_kwargs"]["layout"][0], str), "layout name must be a string"
                    assert isinstance(d["quant_type_kwargs"]["layout"][1], dict), "layout kwargs must be a dict"
                else:
                    raise ValueError("layout must be a list")
        else:
            # Handle AOBaseConfig serialization
            from torchao.core.config import config_to_dict

            # For now we assume there is 1 config per Transformer, however in the future
            # We may want to support a config per fqn.
            d["quant_type"] = {"default": config_to_dict(self.quant_type)}

        return d

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):

        if not is_torchao_version(">", "0.9.0"):
            raise NotImplementedError("TorchAoConfig requires torchao > 0.9.0 for construction from dict")
        config_dict = config_dict.copy()
        quant_type = config_dict.pop("quant_type")

        if isinstance(quant_type, str):
            return cls(quant_type=quant_type, **config_dict)
        # Check if we only have one key which is "default"
        # In the future we may update this
        assert len(quant_type) == 1 and "default" in quant_type, (
            "Expected only one key 'default' in quant_type dictionary"
        )
        quant_type = quant_type["default"]

        # Deserialize quant_type if needed
        from torchao.core.config import config_from_dict

        quant_type = config_from_dict(quant_type)

        return cls(quant_type=quant_type, **config_dict)

    @classmethod
    def _get_torchao_quant_type_to_method(cls):


        if is_torchao_available():
            # TODO(aryan): Support autoquant and sparsify
            from torchao.quantization import (
                float8_dynamic_activation_float8_weight,
                float8_static_activation_float8_weight,
                float8_weight_only,
                int4_weight_only,
                int8_dynamic_activation_int4_weight,
                int8_dynamic_activation_int8_weight,
                int8_weight_only,
                uintx_weight_only,
            )

            if is_torchao_version("<=", "0.14.1"):
                from torchao.quantization import fpx_weight_only
            # TODO(aryan): Add a note on how to use PerAxis and PerGroup observers
            from torchao.quantization.observer import PerRow, PerTensor

            def generate_float8dq_types(dtype: torch.dtype):
                name = "e5m2" if dtype == torch.float8_e5m2 else "e4m3"
                types = {}

                for granularity_cls in [PerTensor, PerRow]:
                    # Note: Activation and Weights cannot have different granularities
                    granularity_name = "tensor" if granularity_cls is PerTensor else "row"
                    types[f"float8dq_{name}_{granularity_name}"] = partial(
                        float8_dynamic_activation_float8_weight,
                        activation_dtype=dtype,
                        weight_dtype=dtype,
                        granularity=(granularity_cls(), granularity_cls()),
                    )

                return types

            def generate_fpx_quantization_types(bits: int):
                if is_torchao_version("<=", "0.14.1"):
                    types = {}

                    for ebits in range(1, bits):
                        mbits = bits - ebits - 1
                        types[f"fp{bits}_e{ebits}m{mbits}"] = partial(fpx_weight_only, ebits=ebits, mbits=mbits)

                    non_sign_bits = bits - 1
                    default_ebits = (non_sign_bits + 1) // 2
                    default_mbits = non_sign_bits - default_ebits
                    types[f"fp{bits}"] = partial(fpx_weight_only, ebits=default_ebits, mbits=default_mbits)

                    return types
                else:
                    raise ValueError("Floating point X-bit quantization is not supported in torchao >= 0.15.0")

            INT4_QUANTIZATION_TYPES = {
                # int4 weight + bfloat16/float16 activation
                "int4wo": int4_weight_only,
                "int4_weight_only": int4_weight_only,
                # int4 weight + int8 activation
                "int4dq": int8_dynamic_activation_int4_weight,
                "int8_dynamic_activation_int4_weight": int8_dynamic_activation_int4_weight,
            }

            INT8_QUANTIZATION_TYPES = {
                # int8 weight + bfloat16/float16 activation
                "int8wo": int8_weight_only,
                "int8_weight_only": int8_weight_only,
                # int8 weight + int8 activation
                "int8dq": int8_dynamic_activation_int8_weight,
                "int8_dynamic_activation_int8_weight": int8_dynamic_activation_int8_weight,
            }

            # TODO(aryan): handle torch 2.2/2.3
            FLOATX_QUANTIZATION_TYPES = {
                # float8_e5m2 weight + bfloat16/float16 activation
                "float8wo": partial(float8_weight_only, weight_dtype=torch.float8_e5m2),
                "float8_weight_only": float8_weight_only,
                "float8wo_e5m2": partial(float8_weight_only, weight_dtype=torch.float8_e5m2),
                # float8_e4m3 weight + bfloat16/float16 activation
                "float8wo_e4m3": partial(float8_weight_only, weight_dtype=torch.float8_e4m3fn),
                # float8_e5m2 weight + float8 activation (dynamic)
                "float8dq": float8_dynamic_activation_float8_weight,
                "float8_dynamic_activation_float8_weight": float8_dynamic_activation_float8_weight,
                # ===== Matrix multiplication is n...
                # However, changing activation_dtype=torch.float8_e4m3 might work here =====
                # "float8dq_e5m2": partial(
                #     float8_dynamic_activation_float8_weight,
                #     activation_dtype=torch.float8_e5m2,
                #     weight_dtype=torch.float8_e5m2,
                # ),
                # **generate_float8dq_types(torch.float8_e5m2),
                # ===== =====
                # float8_e4m3 weight + float8 activation (dynamic)
                "float8dq_e4m3": partial(
                    float8_dynamic_activation_float8_weight,
                    activation_dtype=torch.float8_e4m3fn,
                    weight_dtype=torch.float8_e4m3fn,
                ),
                **generate_float8dq_types(torch.float8_e4m3fn),
                # float8 weight + float8 activation (static)
                "float8_static_activation_float8_weight": float8_static_activation_float8_weight,
            }

            if is_torchao_version("<=", "0.14.1"):
                FLOATX_QUANTIZATION_TYPES.update(generate_fpx_quantization_types(3))
                FLOATX_QUANTIZATION_TYPES.update(generate_fpx_quantization_types(4))
                FLOATX_QUANTIZATION_TYPES.update(generate_fpx_quantization_types(5))
                FLOATX_QUANTIZATION_TYPES.update(generate_fpx_quantization_types(6))
                FLOATX_QUANTIZATION_TYPES.update(generate_fpx_quantization_types(7))

            UINTX_QUANTIZATION_DTYPES = {
                "uintx_weight_only": uintx_weight_only,
                "uint1wo": partial(uintx_weight_only, dtype=torch.uint1),
                "uint2wo": partial(uintx_weight_only, dtype=torch.uint2),
                "uint3wo": partial(uintx_weight_only, dtype=torch.uint3),
                "uint4wo": partial(uintx_weight_only, dtype=torch.uint4),
                "uint5wo": partial(uintx_weight_only, dtype=torch.uint5),
                "uint6wo": partial(uintx_weight_only, dtype=torch.uint6),
                "uint7wo": partial(uintx_weight_only, dtype=torch.uint7),
                # "uint8wo": partial(uintx_weight_...
            }

            QUANTIZATION_TYPES = {}
            QUANTIZATION_TYPES.update(INT4_QUANTIZATION_TYPES)
            QUANTIZATION_TYPES.update(INT8_QUANTIZATION_TYPES)
            QUANTIZATION_TYPES.update(UINTX_QUANTIZATION_DTYPES)

            if cls._is_xpu_or_cuda_capability_atleast_8_9():
                QUANTIZATION_TYPES.update(FLOATX_QUANTIZATION_TYPES)

            return QUANTIZATION_TYPES
        else:
            raise ValueError(
                "TorchAoConfig requires torchao to be installed, please install with `pip install torchao`"
            )

    @staticmethod
    def _is_xpu_or_cuda_capability_atleast_8_9() -> bool:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major == 8:
                return minor >= 9
            return major >= 9
        elif torch.xpu.is_available():
            return True
        else:
            raise RuntimeError("TorchAO requires a CUDA compatible GPU or Intel XPU and installation of PyTorch.")

    def get_apply_tensor_subclass(self):

        if not isinstance(self.quant_type, str):
            return self.quant_type
        else:
            methods = self._get_torchao_quant_type_to_method()
            quant_type_kwargs = self.quant_type_kwargs.copy()
            if (
                not torch.cuda.is_available()
                and is_torchao_available()
                and self.quant_type == "int4_weight_only"
                and version.parse(importlib.metadata.version("torchao")) >= version.parse("0.8.0")
                and quant_type_kwargs.get("layout", None) is None
            ):
                if torch.xpu.is_available():
                    if version.parse(importlib.metadata.version("torchao")) >= version.parse(
                        "0.11.0"
                    ) and version.parse(importlib.metadata.version("torch")) > version.parse("2.7.9"):
                        from torchao.dtypes import Int4XPULayout
                        from torchao.quantization.quant_primitives import ZeroPointDomain

                        quant_type_kwargs["layout"] = Int4XPULayout()
                        quant_type_kwargs["zero_point_domain"] = ZeroPointDomain.INT
                    else:
                        raise ValueError(
                            "TorchAoConfig requires torchao >= 0.11.0 and torch >= 2.8.0 for XPU support. Please upgrade the version or use run on CPU with the cpu version pytorch."
                        )
                else:
                    from torchao.dtypes import Int4CPULayout

                    quant_type_kwargs["layout"] = Int4CPULayout()

            return methods[self.quant_type](**quant_type_kwargs)

    def __repr__(self):

        config_dict = self.to_dict()
        return (
            f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True, cls=TorchAoJSONEncoder)}\n"
        )

@dataclass
class QuantoConfig(QuantizationConfigMixin):
    
    class QuantoConfig(QuantizationConfigMixin):


    def __init__(
        self,
        weights_dtype: str = "int8",
        modules_to_not_convert: Optional[List[str]] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.QUANTO
        self.weights_dtype = weights_dtype
        self.modules_to_not_convert = modules_to_not_convert

        self.post_init()

    def post_init(self):

        accepted_weights = ["float8", "int8", "int4", "int2"]
        if self.weights_dtype not in accepted_weights:
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights_dtype}")

@dataclass
class NVIDIAModelOptConfig(QuantizationConfigMixin):
    
    class NVIDIAModelOptConfig(QuantizationConfigMixin):


    quanttype_to_numbits = {
        "FP8": (4, 3),
        "INT8": 8,
        "INT4": 4,
        "NF4": 4,
        "NVFP4": (2, 1),
    }
    quanttype_to_scalingbits = {
        "NF4": 8,
        "NVFP4": (4, 3),
    }

    def __init__(
        self,
        quant_type: str,
        modules_to_not_convert: Optional[List[str]] = None,
        weight_only: bool = True,
        channel_quantize: Optional[int] = None,
        block_quantize: Optional[int] = None,
        scale_channel_quantize: Optional[int] = None,
        scale_block_quantize: Optional[int] = None,
        algorithm: str = "max",
        forward_loop: Optional[Callable] = None,
        modelopt_config: Optional[dict] = None,
        disable_conv_quantization: bool = False,
        **kwargs,
    ) -> None:
        self.quant_method = QuantizationMethod.MODELOPT
        self._normalize_quant_type(quant_type)
        self.modules_to_not_convert = modules_to_not_convert
        self.weight_only = weight_only
        self.channel_quantize = channel_quantize
        self.block_quantize = block_quantize
        self.calib_cfg = {
            "method": algorithm,
            # add more options here if needed
        }
        self.forward_loop = forward_loop
        self.scale_channel_quantize = scale_channel_quantize
        self.scale_block_quantize = scale_block_quantize
        self.modelopt_config = self.get_config_from_quant_type() if not modelopt_config else modelopt_config
        self.disable_conv_quantization = disable_conv_quantization

    def check_model_patching(self, operation: str = "loading"):
        # ModelOpt imports diffusers internally. This is here to prevent circular imports
        from modelopt.torch.opt.plugins.huggingface import _PATCHED_CLASSES

        if len(_PATCHED_CLASSES) == 0:
            warning_msg = (
                f"Not {operation} weights in modelopt format. This might cause unreliable behavior."
                "Please make sure to run the following code before loading/saving model weights:\n\n"
                "    from modelopt.torch.opt import enable_huggingface_checkpointing\n"
                "    enable_huggingface_checkpointing()\n"
            )
            warnings.warn(warning_msg)

    def _normalize_quant_type(self, quant_type: str) -> str:

        parts = quant_type.split("_")
        w_type = parts[0]
        act_type = parts[1] if len(parts) > 1 else None
        if len(parts) > 2:
            logger.warning(f"Quantization type {quant_type} is not supported. Picking FP8_INT8 as default")
            w_type = "FP8"
            act_type = None
        else:
            if w_type not in NVIDIAModelOptConfig.quanttype_to_numbits:
                logger.warning(f"Weight Quantization type {w_type} is not supported. Picking FP8 as default")
                w_type = "FP8"
            if act_type is not None and act_type not in NVIDIAModelOptConfig.quanttype_to_numbits:
                logger.warning(f"Activation Quantization type {act_type} is not supported. Picking INT8 as default")
                act_type = None
        self.quant_type = w_type + ("_" + act_type if act_type is not None else "")

    def get_config_from_quant_type(self) -> Dict[str, Any]:

        import modelopt.torch.quantization as mtq

        BASE_CONFIG = {
            "quant_cfg": {
                "*weight_quantizer": {"fake_quant": False},
                "*input_quantizer": {},
                "*output_quantizer": {"enable": False},
                "*q_bmm_quantizer": {},
                "*k_bmm_quantizer": {},
                "*v_bmm_quantizer": {},
                "*softmax_quantizer": {},
                **mtq.config._default_disabled_quantizer_cfg,
            },
            "algorithm": self.calib_cfg,
        }

        quant_cfg = BASE_CONFIG["quant_cfg"]
        if self.weight_only:
            for k in quant_cfg:
                if "*weight_quantizer" not in k and not quant_cfg[k]:
                    quant_cfg[k]["enable"] = False

        parts = self.quant_type.split("_")
        w_type = parts[0]
        act_type = parts[1].replace("A", "") if len(parts) > 1 else None
        for k in quant_cfg:
            if k not in mtq.config._default_disabled_quantizer_cfg and "enable" not in quant_cfg[k]:
                if k == "*input_quantizer":
                    if act_type is not None:
                        quant_cfg[k]["num_bits"] = NVIDIAModelOptConfig.quanttype_to_numbits[act_type]
                    continue
                quant_cfg[k]["num_bits"] = NVIDIAModelOptConfig.quanttype_to_numbits[w_type]

        if self.block_quantize is not None and self.channel_quantize is not None:
            quant_cfg["*weight_quantizer"]["block_sizes"] = {self.channel_quantize: self.block_quantize}
            quant_cfg["*input_quantizer"]["block_sizes"] = {
                self.channel_quantize: self.block_quantize,
                "type": "dynamic",
            }
        elif self.channel_quantize is not None:
            quant_cfg["*weight_quantizer"]["axis"] = self.channel_quantize
            quant_cfg["*input_quantizer"]["axis"] = self.channel_quantize
            quant_cfg["*input_quantizer"]["type"] = "dynamic"

        # Only fixed scaling sizes are supported for now in modelopt
        if self.scale_channel_quantize is not None and self.scale_block_quantize is not None:
            if w_type in NVIDIAModelOptConfig.quanttype_to_scalingbits:
                quant_cfg["*weight_quantizer"]["block_sizes"].update(
                    {
                        "scale_bits": NVIDIAModelOptConfig.quanttype_to_scalingbits[w_type],
                        "scale_block_sizes": {self.scale_channel_quantize: self.scale_block_quantize},
                    }
                )
            if act_type and act_type in NVIDIAModelOptConfig.quanttype_to_scalingbits:
                quant_cfg["*input_quantizer"]["block_sizes"].update(
                    {
                        "scale_bits": NVIDIAModelOptConfig.quanttype_to_scalingbits[act_type],
                        "scale_block_sizes": {self.scale_channel_quantize: self.scale_block_quantize},
                    }
                )

        return BASE_CONFIG
