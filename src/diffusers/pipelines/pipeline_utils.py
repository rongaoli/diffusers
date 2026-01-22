# coding=utf-8
import fnmatch
import importlib
import inspect
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin

import httpx
import numpy as np
import PIL.Image
import requests
import torch
from huggingface_hub import (
    DDUFEntry,
    ModelCard,
    create_repo,
    hf_hub_download,
    model_info,
    read_dduf_file,
    snapshot_download,
)
from huggingface_hub.utils import HfHubHTTPError, OfflineModeIsEnabled, validate_hf_hub_args
from packaging import version
from tqdm.auto import tqdm
from typing_extensions import Self

from .. import __version__
from ..configuration_utils import ConfigMixin
from ..models import AutoencoderKL
from ..models.attention_processor import FusedAttnProcessor2_0
from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, ModelMixin
from ..quantizers import PipelineQuantizationConfig
from ..quantizers.bitsandbytes.utils import _check_bnb_status
from ..schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from ..utils import (
    CONFIG_NAME,
    DEPRECATED_REVISION_ARGS,
    BaseOutput,
    PushToHubMixin,
    _get_detailed_type,
    _is_valid_type,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    is_bitsandbytes_version,
    is_hpu_available,
    is_torch_npu_available,
    is_torch_version,
    is_transformers_version,
    logging,
    numpy_to_pil,
)
from ..utils.distributed_utils import is_torch_dist_rank_zero
from ..utils.hub_utils import _check_legacy_sharding_variant_format, load_or_create_model_card, populate_model_card
from ..utils.torch_utils import empty_device_cache, get_device, is_compiled_module

if is_torch_npu_available():
    import torch_npu  # noqa: F401

from .pipeline_loading_utils import (
    ALL_IMPORTABLE_CLASSES,
    CONNECTED_PIPES_KEYS,
    CUSTOM_PIPELINE_FILE_NAME,
    LOADABLE_CLASSES,
    _download_dduf_file,
    _fetch_class_library_tuple,
    _get_custom_components_and_folders,
    _get_custom_pipeline_class,
    _get_final_device_map,
    _get_ignore_patterns,
    _get_pipeline_class,
    _identify_model_variants,
    _maybe_raise_error_for_incorrect_transformers,
    _maybe_raise_warning_for_inpainting,
    _maybe_warn_for_wrong_component_in_quant_config,
    _resolve_custom_pipeline_and_cls,
    _unwrap_model,
    _update_init_kwargs_with_connected_pipeline,
    filter_model_files,
    load_sub_model,
    maybe_raise_or_warn,
    variant_compatible_siblings,
    warn_deprecated_model_variant,
)

if is_accelerate_available():
    import accelerate

LIBRARIES = []
for library in LOADABLE_CLASSES:
    LIBRARIES.append(library)

SUPPORTED_DEVICE_MAP = ["balanced"] + [get_device()]

logger = logging.get_logger(__name__)

@dataclass
class ImagePipelineOutput(BaseOutput):


    images: Union[List[PIL.Image.Image], np.ndarray]

@dataclass
class AudioPipelineOutput(BaseOutput):


    audios: np.ndarray

class DeprecatedPipelineMixin:


    # Override this in the inheriting class to spe...
    _last_supported_version = None

    def __init__(self, *args, **kwargs):
        # Get the class name for the warning message
        class_name = self.__class__.__name__

        # Get the last supported version or use the current version if not specified
        version_info = getattr(self.__class__, "_last_supported_version", __version__)

        # Raise a warning that this pipeline is deprecated
        logger.warning(
            f"The {class_name} has been deprecated and will not receive bug fixes or feature updates after Diffusers version {version_info}. "
        )

        # Call the parent class's __init__ method
        super().__init__(*args, **kwargs)

class DiffusionPipeline(ConfigMixin, PushToHubMixin):
    class to spe...
    _last_supported_version = None

    def __init__(self, *args, **kwargs):
        # Get the class name for the warning message
        class_name = self.__class__.__name__

        # Get the last supported version or use the current version if not specified
        version_info = getattr(self.__class__, "_last_supported_version", __version__)

        # Raise a warning that this pipeline is deprecated
        logger.warning(
            f"The {class_name} has been deprecated and will not receive bug fixes or feature updates after Diffusers version {version_info}. "
        )

        # Call the parent class's __init__ method
        super().__init__(*args, **kwargs)

class DiffusionPipeline(ConfigMixin, PushToHubMixin):


    config_name = "model_index.json"
    model_cpu_offload_seq = None
    hf_device_map = None
    _optional_components = []
    _exclude_from_cpu_offload = []
    _load_connected_pipes = False
    _is_onnx = False

    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # retrieve library
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                register_dict = {name: (None, None)}
            else:
                library, class_name = _fetch_class_library_tuple(module)
                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def __setattr__(self, name: str, value: Any):
        if name in self.__dict__ and hasattr(self.config, name):
            # We need to overwrite the config if name exists in config
            if isinstance(getattr(self.config, name), (tuple, list)):
                if value is not None and self.config[name][0] is not None:
                    class_library_tuple = _fetch_class_library_tuple(value)
                else:
                    class_library_tuple = (None, None)

                self.register_to_config(**{name: class_library_tuple})
            else:
                self.register_to_config(**{name: value})

        super().__setattr__(name, value)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Optional[Union[int, str]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):

        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name", None)
        model_index_dict.pop("_diffusers_version", None)
        model_index_dict.pop("_module", None)
        model_index_dict.pop("_name_or_path", None)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        expected_modules, optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)}
        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            # Dynamo wraps the original model in a private class.
            # I didn't find a public API to get the original class.
            if is_compiled_module(sub_model):
                sub_model = _unwrap_model(sub_model)
                model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                if library_name in sys.modules:
                    library = importlib.import_module(library_name)
                else:
                    logger.info(
                        f"{library_name} is not installed. Cannot save {pipeline_component_name} as {library_classes} from {library_name}"
                    )

                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable b...
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            if save_method_name is None:
                logger.warning(
                    f"self.{pipeline_component_name}={sub_model} of type {type(sub_model)} cannot be saved."
                )
                # make sure that unsaveable components are not tried to be loaded afterward
                self.register_to_config(**{pipeline_component_name: (None, None)})
                continue

            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = "safe_serialization" in save_method_signature.parameters
            save_method_accept_variant = "variant" in save_method_signature.parameters
            save_method_accept_max_shard_size = "max_shard_size" in save_method_signature.parameters

            save_kwargs = {}
            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant
            if save_method_accept_max_shard_size and max_shard_size is not None:
                # max_shard_size is expected to not be None in ModelMixin
                save_kwargs["max_shard_size"] = max_shard_size

            save_method(os.path.join(save_directory, pipeline_component_name), **save_kwargs)

        # finally save the config
        self.save_config(save_directory)

        if push_to_hub:
            # Create a new empty model card and eventually tag it
            model_card = load_or_create_model_card(repo_id, token=token, is_pipeline=True)
            model_card = populate_model_card(model_card)
            model_card.save(os.path.join(save_directory, "README.md"))

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    def to(self, *args, **kwargs) -> Self:

        Performs Pipeline dtype and/or device conversion. A torch.dtype and torch.device are inferred from the
        arguments of `self.to(*args, **kwargs).`

        > [!TIP] > If the pipeline already has the correct torch.dtype and torch.device, then it is returned as is.
        Otherwise, > the returned pipeline is a copy of self with the desired torch.dtype and torch.device.

        Here are the ways to call `to`:

        - `to(dtype, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
        - `to(device, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
        - `to(device=None, dtype=None, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the
          specified [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) and
          [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
                [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
            device (`torch.Device`, *optional*):
                Returns a pipeline with the specified
                [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
            silence_dtype_warnings (`str`, *optional*, defaults to `False`):
                Whether to omit warnings if the target `dtype` is not compatible with the target `device`.
