import os
from typing import Optional, Union

from huggingface_hub.utils import validate_hf_hub_args

from ..configuration_utils import ConfigMixin
from ..utils import logging
from ..utils.dynamic_modules_utils import get_class_from_dynamic_module, resolve_trust_remote_code

logger = logging.get_logger(__name__)

class AutoModel(ConfigMixin):
    config_name = "config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path: Optional[Union[str, os.PathLike]] = None, **kwargs):
        class AutoModel(ConfigMixin):
    config_name = "config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path: Optional[Union[str, os.PathLike]] = None, **kwargs):

        subfolder = kwargs.pop("subfolder", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)

        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "token",
        ]
        hub_kwargs = {name: kwargs.pop(name, None) for name in hub_kwargs_names}

        # load_config_kwargs uses the same hub kwargs minus subfolder and resume_download
        load_config_kwargs = {k: v for k, v in hub_kwargs.items() if k not in ["subfolder"]}

        library = None
        orig_class_name = None

        # Always attempt to fetch model_index.json first
        try:
            cls.config_name = "model_index.json"
            config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)

            if subfolder is not None and subfolder in config:
                library, orig_class_name = config[subfolder]
                load_config_kwargs.update({"subfolder": subfolder})

        except EnvironmentError as e:
            logger.debug(e)

        # Unable to load from model_index.json so fallback to loading from config
        if library is None and orig_class_name is None:
            cls.config_name = "config.json"
            config = cls.load_config(pretrained_model_or_path, subfolder=subfolder, **load_config_kwargs)

            if "_class_name" in config:
                # If we find a class name in the c...
                orig_class_name = config["_class_name"]
                library = "diffusers"
                load_config_kwargs.update({"subfolder": subfolder})
            elif "model_type" in config:
                orig_class_name = "AutoModel"
                library = "transformers"
                load_config_kwargs.update({"subfolder": "" if subfolder is None else subfolder})
            else:
                raise ValueError(f"Couldn't find model associated with the config file at {pretrained_model_or_path}.")

        has_remote_code = "auto_map" in config and cls.__name__ in config["auto_map"]
        trust_remote_code = resolve_trust_remote_code(trust_remote_code, pretrained_model_or_path, has_remote_code)
        if not has_remote_code and trust_remote_code:
            raise ValueError(
                "Selected model repository does not happear to have any custom code or does not have a valid `config.json` file."
            )

        if has_remote_code and trust_remote_code:
            class_ref = config["auto_map"][cls.__name__]
            module_file, class_name = class_ref.split(".")
            module_file = module_file + ".py"
            model_cls = get_class_from_dynamic_module(
                pretrained_model_or_path,
                subfolder=subfolder,
                module_file=module_file,
                class_name=class_name,
                **hub_kwargs,
            )
        else:
            from ..pipelines.pipeline_loading_utils import ALL_IMPORTABLE_CLASSES, get_class_obj_and_candidates

            model_cls, _ = get_class_obj_and_candidates(
                library_name=library,
                class_name=orig_class_name,
                importable_classes=ALL_IMPORTABLE_CLASSES,
                pipelines=None,
                is_pipeline_module=False,
            )

        if model_cls is None:
            raise ValueError(f"AutoModel can't find a model linked to {orig_class_name}.")

        kwargs = {**load_config_kwargs, **kwargs}
        return model_cls.from_pretrained(pretrained_model_or_path, **kwargs)
