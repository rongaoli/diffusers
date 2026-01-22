# coding=utf-8
import importlib
import inspect
import os
from typing import Any, Dict, List, Optional, Union

import flax
import numpy as np
import PIL.Image
from flax.core.frozen_dict import FrozenDict
from huggingface_hub import create_repo, snapshot_download
from huggingface_hub.utils import validate_hf_hub_args
from PIL import Image
from tqdm.auto import tqdm

from ..configuration_utils import ConfigMixin
from ..models.modeling_flax_utils import FLAX_WEIGHTS_NAME, FlaxModelMixin
from ..schedulers.scheduling_utils_flax import SCHEDULER_CONFIG_NAME, FlaxSchedulerMixin
from ..utils import (
    CONFIG_NAME,
    BaseOutput,
    PushToHubMixin,
    http_user_agent,
    is_transformers_available,
    logging,
)

if is_transformers_available():
    from transformers import FlaxPreTrainedModel

INDEX_FILE = "diffusion_flax_model.bin"

logger = logging.get_logger(__name__)

LOADABLE_CLASSES = {
    "diffusers": {
        "FlaxModelMixin": ["save_pretrained", "from_pretrained"],
        "FlaxSchedulerMixin": ["save_pretrained", "from_pretrained"],
        "FlaxDiffusionPipeline": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "FlaxPreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])

def import_flax_or_no_model(module, class_name):
    try:
        # 1. First make sure that if a Flax object is present, import this one
        class_obj = getattr(module, "Flax" + class_name)
    except AttributeError:
        # 2. If this doesn't work, it's not a model and we don't append "Flax"
        class_obj = getattr(module, class_name)
    except AttributeError:
        raise ValueError(f"Neither Flax{class_name} nor {class_name} exist in {module}")

    return class_obj

@flax.struct.dataclass
class FlaxImagePipelineOutput(BaseOutput):


    images: Union[List[PIL.Image.Image], np.ndarray]

class FlaxDiffusionPipeline(ConfigMixin, PushToHubMixin):


    config_name = "model_index.json"

    def register_modules(self, **kwargs):
        # import it here to avoid circular import
        from diffusers import pipelines

        for name, module in kwargs.items():
            if module is None:
                register_dict = {name: (None, None)}
            else:
                # retrieve library
                library = module.__module__.split(".")[0]

                # check if the module is a pipeline module
                pipeline_dir = module.__module__.split(".")[-2]
                path = module.__module__.split(".")
                is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

                # if library is not in LOADABLE_CLASSES, then it is a custom module.
                # Or if it's a pipeline module, then the module is inside the pipeline
                # folder so we set the library to module name.
                if library not in LOADABLE_CLASSES or is_pipeline_module:
                    library = pipeline_dir

                # retrieve class_name
                class_name = module.__class__.__name__

                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        params: Union[Dict, FrozenDict],
        push_to_hub: bool = False,
        **kwargs,
    ):
        # TODO: handle inference_state

        self.save_config(save_directory)

        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name")
        model_index_dict.pop("_diffusers_version")
        model_index_dict.pop("_module", None)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            if sub_model is None:
                # edge case for saving a pipeline with safety_checker=None
                continue

            model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                library = importlib.import_module(library_name)
                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable b...
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            save_method = getattr(sub_model, save_method_name)
            expects_params = "params" in set(inspect.signature(save_method).parameters.keys())

            if expects_params:
                save_method(
                    os.path.join(save_directory, pipeline_component_name), params=params[pipeline_component_name]
                )
            else:
                save_method(os.path.join(save_directory, pipeline_component_name))

            if push_to_hub:
                self._upload_folder(
                    save_directory,
                    repo_id,
                    token=token,
                    commit_message=commit_message,
                    create_pr=create_pr,
                )

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        class implements both a save and loading method. The pipeline is easily reloaded using the
        [`~FlaxDiffusionPipeline.from_pretrained`] class method.

        使用示例见文档

        logger.warning(
            "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We "
            "recommend migrating to PyTorch classes or pinning your version of Diffusers."
        )

        cache_dir = kwargs.pop("cache_dir", None)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_pt = kwargs.pop("from_pt", False)
        use_memory_efficient_attention = kwargs.pop("use_memory_efficient_attention", False)
        split_head_dim = kwargs.pop("split_head_dim", False)
        dtype = kwargs.pop("dtype", None)

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            config_dict = cls.load_config(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
            # make sure we only download sub-folders and `diffusers` filenames
            folder_names = [k for k in config_dict.keys() if not k.startswith("_")]
            allow_patterns = [os.path.join(k, "*") for k in folder_names]
            allow_patterns += [FLAX_WEIGHTS_NAME, SCHEDULER_CONFIG_NAME, CONFIG_NAME, cls.config_name]

            ignore_patterns = ["*.bin", "*.safetensors"] if not from_pt else []
            ignore_patterns += ["*.onnx", "*.onnx_data", "*.xml", "*.pb"]

            if cls != FlaxDiffusionPipeline:
                requested_pipeline_class = cls.__name__
            else:
                requested_pipeline_class = config_dict.get("_class_name", cls.__name__)
                requested_pipeline_class = (
                    requested_pipeline_class
                    if requested_pipeline_class.startswith("Flax")
                    else "Flax" + requested_pipeline_class
                )

            user_agent = {"pipeline_class": requested_pipeline_class}
            user_agent = http_user_agent(user_agent)

            # download all allow_patterns
            cached_folder = snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_agent=user_agent,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        config_dict = cls.load_config(cached_folder)

        # 2. Load the pipeline class, if using custom module then load it from the hub
        # if we load from explicit class, let's use it
        if cls != FlaxDiffusionPipeline:
            pipeline_class = cls
        else:
            diffusers_module = importlib.import_module(cls.__module__.split(".")[0])
            class_name = (
                config_dict["_class_name"]
                if config_dict["_class_name"].startswith("Flax")
                else "Flax" + config_dict["_class_name"]
            )
            pipeline_class = getattr(diffusers_module, class_name)

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        # define init kwargs
        init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in init_dict}
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # Throw nice warnings / errors for fast accelerate loading
        if len(unused_kwargs) > 0:
            logger.warning(
                f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
            )

        # inference_params
        params = {}

        # import it here to avoid circular import
        from diffusers import pipelines

        # 3. Load each module in the pipeline
        for name, (library_name, class_name) in init_dict.items():
            if class_name is None:
                # edge case for when the pipeline was saved with safety_checker=None
                init_kwargs[name] = None
                continue

            is_pipeline_module = hasattr(pipelines, library_name)
            loaded_sub_model = None
            sub_model_should_be_defined = True

            # if the model is in a pipeline module, then we load it from the pipeline
            if name in passed_class_obj:
                # 1. check that passed_class_obj has correct parent class
                if not is_pipeline_module:
                    library = importlib.import_module(library_name)
                    class_obj = getattr(library, class_name)
                    importable_classes = LOADABLE_CLASSES[library_name]
                    class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

                    expected_class_obj = None
                    for class_name, class_candidate in class_candidates.items():
                        if class_candidate is not None and issubclass(class_obj, class_candidate):
                            expected_class_obj = class_candidate

                    if not issubclass(passed_class_obj[name].__class__, expected_class_obj):
                        raise ValueError(
                            f"{passed_class_obj[name]} is of type: {type(passed_class_obj[name])}, but should be"
                            f" {expected_class_obj}"
                        )
                elif passed_class_obj[name] is None:
                    logger.warning(
                        f"You have passed `None` for {name} to disable its functionality in {pipeline_class}. Note"
                        f" that this might lead to problems when using {pipeline_class} and is not recommended."
                    )
                    sub_model_should_be_defined = False
                else:
                    logger.warning(
                        f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
                        " has the correct type"
                    )

                # set passed class object
                loaded_sub_model = passed_class_obj[name]
            elif is_pipeline_module:
                pipeline_module = getattr(pipelines, library_name)
                class_obj = import_flax_or_no_model(pipeline_module, class_name)

                importable_classes = ALL_IMPORTABLE_CLASSES
                class_candidates = dict.fromkeys(importable_classes.keys(), class_obj)
            else:
                # else we just import it from the library.
                library = importlib.import_module(library_name)
                class_obj = import_flax_or_no_model(library, class_name)

                importable_classes = LOADABLE_CLASSES[library_name]
                class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

            if loaded_sub_model is None and sub_model_should_be_defined:
                load_method_name = None
                for class_name, class_candidate in class_candidates.items():
                    if class_candidate is not None and issubclass(class_obj, class_candidate):
                        load_method_name = importable_classes[class_name][1]

                load_method = getattr(class_obj, load_method_name)

                # check if the module is in a subdirectory
                if os.path.isdir(os.path.join(cached_folder, name)):
                    loadable_folder = os.path.join(cached_folder, name)
                else:
                    loaded_sub_model = cached_folder

                if issubclass(class_obj, FlaxModelMixin):
                    loaded_sub_model, loaded_params = load_method(
                        loadable_folder,
                        from_pt=from_pt,
                        use_memory_efficient_attention=use_memory_efficient_attention,
                        split_head_dim=split_head_dim,
                        dtype=dtype,
                    )
                    params[name] = loaded_params
                elif is_transformers_available() and issubclass(class_obj, FlaxPreTrainedModel):
                    if from_pt:
                        # TODO(Suraj): Fix this in Transformers. We should be able to use `_do_init=False` here
                        loaded_sub_model = load_method(loadable_folder, from_pt=from_pt)
                        loaded_params = loaded_sub_model.params
                        del loaded_sub_model._params
                    else:
                        loaded_sub_model, loaded_params = load_method(loadable_folder, _do_init=False)
                    params[name] = loaded_params
                elif issubclass(class_obj, FlaxSchedulerMixin):
                    loaded_sub_model, scheduler_state = load_method(loadable_folder)
                    params[name] = scheduler_state
                else:
                    loaded_sub_model = load_method(loadable_folder)

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        # 4. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())

        if len(missing_modules) > 0 and missing_modules <= set(passed_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        model = pipeline_class(**init_kwargs, dtype=dtype)
        return model, params

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        return expected_modules, optional_parameters

    @property
    def components(self) -> Dict[str, Any]:

        The `self.components` property can be useful to run different pipelines with the same weights and
        configurations to not have to re-allocate memory.
        使用示例见文档
