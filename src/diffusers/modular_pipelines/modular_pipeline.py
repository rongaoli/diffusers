import importlib
import inspect
import os
import traceback
import warnings
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import create_repo
from huggingface_hub.utils import validate_hf_hub_args
from tqdm.auto import tqdm
from typing_extensions import Self

from ..configuration_utils import ConfigMixin, FrozenDict
from ..pipelines.pipeline_loading_utils import _fetch_class_library_tuple, simple_get_class_obj
from ..utils import PushToHubMixin, is_accelerate_available, logging
from ..utils.dynamic_modules_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ..utils.hub_utils import load_or_create_model_card, populate_model_card
from .components_manager import ComponentsManager
from .modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    InsertableDict,
    OutputParam,
    format_components,
    format_configs,
    make_doc_string,
)

if is_accelerate_available():
    import accelerate

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# map regular pipeline to modular pipeline class name
MODULAR_PIPELINE_MAPPING = OrderedDict(
    [
        ("stable-diffusion-xl", "StableDiffusionXLModularPipeline"),
        ("wan", "WanModularPipeline"),
        ("flux", "FluxModularPipeline"),
        ("flux-kontext", "FluxKontextModularPipeline"),
        ("flux2", "Flux2ModularPipeline"),
        ("qwenimage", "QwenImageModularPipeline"),
        ("qwenimage-edit", "QwenImageEditModularPipeline"),
        ("qwenimage-edit-plus", "QwenImageEditPlusModularPipeline"),
        ("qwenimage-layered", "QwenImageLayeredModularPipeline"),
        ("z-image", "ZImageModularPipeline"),
    ]
)

@dataclass
class PipelineState:


    values: Dict[str, Any] = field(default_factory=dict)
    kwargs_mapping: Dict[str, List[str]] = field(default_factory=dict)

    def set(self, key: str, value: Any, kwargs_type: str = None):

        self.values[key] = value

        if kwargs_type is not None:
            if kwargs_type not in self.kwargs_mapping:
                self.kwargs_mapping[kwargs_type] = [key]
            else:
                self.kwargs_mapping[kwargs_type].append(key)

    def get(self, keys: Union[str, List[str]], default: Any = None) -> Union[Any, Dict[str, Any]]:

        if isinstance(keys, str):
            return self.values.get(keys, default)
        return {key: self.values.get(key, default) for key in keys}

    def get_by_kwargs(self, kwargs_type: str) -> Dict[str, Any]:

        value_names = self.kwargs_mapping.get(kwargs_type, [])
        return self.get(value_names)

    def to_dict(self) -> Dict[str, Any]:

        return {**self.__dict__}

    def __getattr__(self, name):

        # Use object.__getattribute__ to avoid infinite recursion during deepcopy
        try:
            values = object.__getattribute__(self, "values")
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        if name in values:
            return values[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self):
        def format_value(v):
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                return f"Tensor(dtype={v.dtype}, shape={v.shape})"
            elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "shape") and hasattr(v[0], "dtype"):
                return f"[Tensor(dtype={v[0].dtype}, shape={v[0].shape}), ...]"
            else:
                return repr(v)

        values_str = "\n".join(f"    {k}: {format_value(v)}" for k, v in self.values.items())
        kwargs_mapping_str = "\n".join(f"    {k}: {v}" for k, v in self.kwargs_mapping.items())

        return f"PipelineState(\n  values={{\n{values_str}\n  }},\n  kwargs_mapping={{\n{kwargs_mapping_str}\n  }}\n)"

@dataclass
class BlockState:


    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key: str):
        # allows block_state["foo"]
        return getattr(self, key, None)

    def __setitem__(self, key: str, value: Any):
        # allows block_state["foo"] = "bar"
        setattr(self, key, value)

    def as_dict(self):


    config_name = "modular_model_index.json"
    hf_device_map = None
    default_blocks_name = None

    # YiYi TODO: add warning for passing multiple ComponentSpec/ConfigSpec with the same name
    def __init__(
        self,
        blocks: Optional[ModularPipelineBlocks] = None,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        components_manager: Optional[ComponentsManager] = None,
        collection: Optional[str] = None,
        modular_config_dict: Optional[Dict[str, Any]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):


        if modular_config_dict is None and config_dict is None and pretrained_model_name_or_path is not None:
            cache_dir = kwargs.pop("cache_dir", None)
            force_download = kwargs.pop("force_download", False)
            proxies = kwargs.pop("proxies", None)
            token = kwargs.pop("token", None)
            local_files_only = kwargs.pop("local_files_only", False)
            revision = kwargs.pop("revision", None)

            load_config_kwargs = {
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "token": token,
                "local_files_only": local_files_only,
                "revision": revision,
            }

            modular_config_dict, config_dict = self._load_pipeline_config(
                pretrained_model_name_or_path, **load_config_kwargs
            )

        if blocks is None:
            if modular_config_dict is not None:
                blocks_class_name = modular_config_dict.get("_blocks_class_name")
            else:
                blocks_class_name = self.get_default_blocks_name(config_dict)
            if blocks_class_name is not None:
                diffusers_module = importlib.import_module("diffusers")
                blocks_class = getattr(diffusers_module, blocks_class_name)
                blocks = blocks_class()
            else:
                logger.warning(f"`blocks` is `None`, no default blocks class found for {self.__class__.__name__}")

        self.blocks = blocks
        self._components_manager = components_manager
        self._collection = collection
        self._component_specs = {spec.name: deepcopy(spec) for spec in self.blocks.expected_components}
        self._config_specs = {spec.name: deepcopy(spec) for spec in self.blocks.expected_configs}

        # update component_specs and config_specs based on modular_model_index.json
        if modular_config_dict is not None:
            for name, value in modular_config_dict.items():
                # all the components in modular_model_index.json are from_pretrained components
                if name in self._component_specs and isinstance(value, (tuple, list)) and len(value) == 3:
                    library, class_name, component_spec_dict = value
                    component_spec = self._dict_to_component_spec(name, component_spec_dict)
                    component_spec.default_creation_method = "from_pretrained"
                    self._component_specs[name] = component_spec

                elif name in self._config_specs:
                    self._config_specs[name].default = value

        # if `modular_config_dict` is None (i.e. `...
        elif config_dict is not None:
            for name, value in config_dict.items():
                if name in self._component_specs and isinstance(value, (tuple, list)) and len(value) == 2:
                    library, class_name = value
                    component_spec_dict = {
                        "repo": pretrained_model_name_or_path,
                        "subfolder": name,
                        "type_hint": (library, class_name),
                    }
                    component_spec = self._dict_to_component_spec(name, component_spec_dict)
                    component_spec.default_creation_method = "from_pretrained"
                    self._component_specs[name] = component_spec
                elif name in self._config_specs:
                    self._config_specs[name].default = value

        if len(kwargs) > 0:
            logger.warning(f"Unexpected input '{kwargs.keys()}' provided. This input will be ignored.")

        register_components_dict = {}
        for name, component_spec in self._component_specs.items():
            if component_spec.default_creation_method == "from_config":
                component = component_spec.create()
            else:
                component = None
            register_components_dict[name] = component
        self.register_components(**register_components_dict)

        default_configs = {}
        for name, config_spec in self._config_specs.items():
            default_configs[name] = config_spec.default
        self.register_to_config(**default_configs)
        self.register_to_config(_blocks_class_name=self.blocks.__class__.__name__ if self.blocks is not None else None)

    @property
    def default_call_parameters(self) -> Dict[str, Any]:
