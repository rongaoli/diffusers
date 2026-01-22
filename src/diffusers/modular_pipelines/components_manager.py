import copy
import time
from collections import OrderedDict
from itertools import combinations
from typing import Any, Dict, List, Optional, Union

import torch

from ..hooks import ModelHook
from ..utils import (
    is_accelerate_available,
    logging,
)
from ..utils.torch_utils import get_device

if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module, remove_hook_from_module
    from accelerate.state import PartialState
    from accelerate.utils import send_to_device
    from accelerate.utils.memory import clear_device_cache
    from accelerate.utils.modeling import convert_file_size_to_int

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class CustomOffloadHook(ModelHook):


    no_grad = False

    def __init__(
        self,
        execution_device: Optional[Union[str, int, torch.device]] = None,
        other_hooks: Optional[List["UserCustomOffloadHook"]] = None,
        offload_strategy: Optional["AutoOffloadStrategy"] = None,
    ):
        self.execution_device = execution_device if execution_device is not None else PartialState().default_device
        self.other_hooks = other_hooks
        self.offload_strategy = offload_strategy
        self.model_id = None

    def set_strategy(self, offload_strategy: "AutoOffloadStrategy"):
        self.offload_strategy = offload_strategy

    def add_other_hook(self, hook: "UserCustomOffloadHook"):

        if self.other_hooks is None:
            self.other_hooks = []
        self.other_hooks.append(hook)

    def init_hook(self, module):
        return module.to("cpu")

    def pre_forward(self, module, *args, **kwargs):
        if module.device != self.execution_device:
            if self.other_hooks is not None:
                hooks_to_offload = [hook for hook in self.other_hooks if hook.model.device == self.execution_device]
                # offload all other hooks
                start_time = time.perf_counter()
                if self.offload_strategy is not None:
                    hooks_to_offload = self.offload_strategy(
                        hooks=hooks_to_offload,
                        model_id=self.model_id,
                        model=module,
                        execution_device=self.execution_device,
                    )
                end_time = time.perf_counter()
                logger.info(
                    f" time taken to apply offload strategy for {self.model_id}: {(end_time - start_time):.2f} seconds"
                )

                for hook in hooks_to_offload:
                    logger.info(
                        f"moving {self.model_id} to {self.execution_device}, offloading {hook.model_id} to cpu"
                    )
                    hook.offload()

                if hooks_to_offload:
                    clear_device_cache()
            module.to(self.execution_device)
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)

class UserCustomOffloadHook:


    def __init__(self, model_id, model, hook):
        self.model_id = model_id
        self.model = model
        self.hook = hook

    def offload(self):
        self.hook.init_hook(self.model)

    def attach(self):
        add_hook_to_module(self.model, self.hook)
        self.hook.model_id = self.model_id

    def remove(self):
        remove_hook_from_module(self.model)
        self.hook.model_id = None

    def add_other_hook(self, hook: "UserCustomOffloadHook"):
        self.hook.add_other_hook(hook)

def custom_offload_with_hook(
    model_id: str,
    model: torch.nn.Module,
    execution_device: Union[str, int, torch.device] = None,
    offload_strategy: Optional["AutoOffloadStrategy"] = None,
):
    hook = CustomOffloadHook(execution_device=execution_device, offload_strategy=offload_strategy)
    user_hook = UserCustomOffloadHook(model_id=model_id, model=model, hook=hook)
    user_hook.attach()
    return user_hook

# this is the class that user can customize to implement their own offload strategy
class AutoOffloadStrategy:


    # YiYi TODO: instead of memory_reserve_margin, we should let user set the maximum_total_models_size to keep on device
    # the actual memory usage would be higher. But it's simpler this way, and can be tested
    def __init__(self, memory_reserve_margin="3GB"):
        self.memory_reserve_margin = convert_file_size_to_int(memory_reserve_margin)

    def __call__(self, hooks, model_id, model, execution_device):
        if len(hooks) == 0:
            return []

        try:
            current_module_size = model.get_memory_footprint()
        except AttributeError:
            raise AttributeError(f"Do not know how to compute memory footprint of `{model.__class__.__name__}.")

        device_type = execution_device.type
        device_module = getattr(torch, device_type, torch.cuda)
        try:
            mem_on_device = device_module.mem_get_info(execution_device.index)[0]
        except AttributeError:
            raise AttributeError(f"Do not know how to obtain obtain memory info for {str(device_module)}.")

        mem_on_device = mem_on_device - self.memory_reserve_margin
        if current_module_size < mem_on_device:
            return []

        min_memory_offload = current_module_size - mem_on_device
        logger.info(f" search for models to offload in order to free up {min_memory_offload / 1024**3:.2f} GB memory")

        # exlucde models that's not currently loaded on the device
        module_sizes = dict(
            sorted(
                {hook.model_id: hook.model.get_memory_footprint() for hook in hooks}.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # YiYi/Dhruv TODO: sort smallest to largest, and offload in that order we would tend to keep the larger models on GPU more often
        def search_best_candidate(module_sizes, min_memory_offload):

            model_ids = list(module_sizes.keys())
            best_candidate = None
            best_size = float("inf")
            for r in range(1, len(model_ids) + 1):
                for candidate_model_ids in combinations(model_ids, r):
                    candidate_size = sum(
                        module_sizes[candidate_model_id] for candidate_model_id in candidate_model_ids
                    )
                    if candidate_size < min_memory_offload:
                        continue
                    else:
                        if best_candidate is None or candidate_size < best_size:
                            best_candidate = candidate_model_ids
                            best_size = candidate_size

            return best_candidate

        best_offload_model_ids = search_best_candidate(module_sizes, min_memory_offload)

        if best_offload_model_ids is None:
            # if no combination is found, meaning...
            logger.warning("no combination of models to offload to cpu is found, offloading all models")
            hooks_to_offload = hooks
        else:
            hooks_to_offload = [hook for hook in hooks if hook.model_id in best_offload_model_ids]

        return hooks_to_offload

# utils for display component info in a readable format
# TODO: move to a different file
def summarize_dict_by_value_and_parts(d: Dict[str, Any]) -> Dict[str, Any]:

    # First group by values - convert lists to tuples to make them hashable
    value_to_keys = {}
    for key, value in d.items():
        value_tuple = tuple(value) if isinstance(value, list) else value
        if value_tuple not in value_to_keys:
            value_to_keys[value_tuple] = []
        value_to_keys[value_tuple].append(key)

    def find_common_prefix(keys: List[str]) -> str:

        if not keys:
            return ""
        if len(keys) == 1:
            return keys[0]

        # Split all keys into parts
        key_parts = [k.split(".") for k in keys]

        # Find how many initial parts are common
        common_length = 0
        for parts in zip(*key_parts):
            if len(set(parts)) == 1:  # All parts at this position are the same
                common_length += 1
            else:
                break

        if common_length == 0:
            return ""

        # Return the common prefix
        return ".".join(key_parts[0][:common_length])

    # Create summary by finding common prefixes for each value group
    summary = {}
    for value_tuple, keys in value_to_keys.items():
        prefix = find_common_prefix(keys)
        if prefix:  # Only add if we found a common prefix
            # Convert tuple back to list if it was originally a list
            value = list(value_tuple) if isinstance(d[keys[0]], list) else value_tuple
            summary[prefix] = value
        else:
            summary[""] = value  # Use empty string if no common prefix

    return summary

class ComponentsManager:


    _available_info_fields = [
        "model_id",
        "added_time",
        "collection",
        "class_name",
        "size_gb",
        "adapters",
        "has_hook",
        "execution_device",
        "ip_adapter",
    ]

    def __init__(self):
        self.components = OrderedDict()
        # YiYi TODO: can remove once confirm we don't need this in mellon
        self.added_time = OrderedDict()  # Store when components were added
        self.collections = OrderedDict()  # collection_name -> set of component_names
        self.model_hooks = None
        self._auto_offload_enabled = False

    def _lookup_ids(
        self,
        name: Optional[str] = None,
        collection: Optional[str] = None,
        load_id: Optional[str] = None,
        components: Optional[OrderedDict] = None,
    ):

        if components is None:
            components = self.components

        if name:
            ids_by_name = set()
            for component_id, component in components.items():
                comp_name = self._id_to_name(component_id)
                if comp_name == name:
                    ids_by_name.add(component_id)
        else:
            ids_by_name = set(components.keys())
        if collection:
            ids_by_collection = set()
            for component_id, component in components.items():
                if component_id in self.collections[collection]:
                    ids_by_collection.add(component_id)
        else:
            ids_by_collection = set(components.keys())
        if load_id:
            ids_by_load_id = set()
            for name, component in components.items():
                if hasattr(component, "_diffusers_load_id") and component._diffusers_load_id == load_id:
                    ids_by_load_id.add(name)
        else:
            ids_by_load_id = set(components.keys())

        ids = ids_by_name.intersection(ids_by_collection).intersection(ids_by_load_id)
        return ids

    @staticmethod
    def _id_to_name(component_id: str):
        return "_".join(component_id.split("_")[:-1])

    def add(self, name: str, component: Any, collection: Optional[str] = None):

        component_id = f"{name}_{id(component)}"
        is_new_component = True

        # check for duplicated components
        for comp_id, comp in self.components.items():
            if comp == component:
                comp_name = self._id_to_name(comp_id)
                if comp_name == name:
                    logger.warning(f"ComponentsManager: component '{name}' already exists as '{comp_id}'")
                    component_id = comp_id
                    is_new_component = False
                    break
                else:
                    logger.warning(
                        f"ComponentsManager: adding component '{name}' as '{component_id}', but it is duplicate of '{comp_id}'"
                        f"To remove a duplicate, call `components_manager.remove('<component_id>')`."
                    )

        # check for duplicated load_id and warn (we do not delete for you)
        if hasattr(component, "_diffusers_load_id") and component._diffusers_load_id != "null":
            components_with_same_load_id = self._lookup_ids(load_id=component._diffusers_load_id)
            components_with_same_load_id = [id for id in components_with_same_load_id if id != component_id]

            if components_with_same_load_id:
                existing = ", ".join(components_with_same_load_id)
                logger.warning(
                    f"ComponentsManager: adding component '{component_id}', but it has duplicate load_id '{component._diffusers_load_id}' with existing components: {existing}. "
                    f"To remove a duplicate, call `components_manager.remove('<component_id>')`."
                )

        # add component to components manager
        self.components[component_id] = component
        self.added_time[component_id] = time.time()

        if collection:
            if collection not in self.collections:
                self.collections[collection] = set()
            if component_id not in self.collections[collection]:
                comp_ids_in_collection = self._lookup_ids(name=name, collection=collection)
                for comp_id in comp_ids_in_collection:
                    logger.warning(
                        f"ComponentsManager: removing existing {name} from collection '{collection}': {comp_id}"
                    )
                    # remove existing component fr...
                    self.remove_from_collection(comp_id, collection)

                self.collections[collection].add(component_id)
                logger.info(
                    f"ComponentsManager: added component '{name}' in collection '{collection}': {component_id}"
                )
        else:
            logger.info(f"ComponentsManager: added component '{name}' as '{component_id}'")

        if self._auto_offload_enabled and is_new_component:
            self.enable_auto_cpu_offload(self._auto_offload_device)

        return component_id

    def remove_from_collection(self, component_id: str, collection: str):

        if collection not in self.collections:
            logger.warning(f"Collection '{collection}' not found in ComponentsManager")
            return
        if component_id not in self.collections[collection]:
            logger.warning(f"Component '{component_id}' not found in collection '{collection}'")
            return
        # remove from the collection
        self.collections[collection].remove(component_id)
        # check if this component is in any other collection
        comp_colls = [coll for coll, comps in self.collections.items() if component_id in comps]
        if not comp_colls:  # only if no other collection contains this component, remove it
            logger.warning(f"ComponentsManager: removing component '{component_id}' from ComponentsManager")
            self.remove(component_id)

    def remove(self, component_id: str = None):

        if component_id not in self.components:
            logger.warning(f"Component '{component_id}' not found in ComponentsManager")
            return

        component = self.components.pop(component_id)
        self.added_time.pop(component_id)

        for collection in self.collections:
            if component_id in self.collections[collection]:
                self.collections[collection].remove(component_id)

        if self._auto_offload_enabled:
            self.enable_auto_cpu_offload(self._auto_offload_device)
        else:
            if isinstance(component, torch.nn.Module):
                component.to("cpu")
            del component
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.xpu.is_available():
                torch.xpu.empty_cache()

    # YiYi TODO: rename to search_components for now, may remove this method
    def search_components(
        self,
        names: Optional[str] = None,
        collection: Optional[str] = None,
        load_id: Optional[str] = None,
        return_dict_with_names: bool = True,
    ):
        Search components by name with simple pattern matching. Optionally filter by collection or load_id.
                - "unet" : match any component with base name "unet" (e.g., unet_123abc)
                - "!unet" : everything except components with base name "unet"
                - "unet*" : anything with base name starting with "unet"
                - "!unet*" : anything with base name NOT starting with "unet"
                - "*unet*" : anything with base name containing "unet"
                - "!*unet*" : anything with base name NOT containing "unet"
                - "refiner|vae|unet" : anything with base name exactly matching "refiner", "vae", or "unet"
                - "!refiner|vae|unet" : anything with base name NOT exactly matching "refiner", "vae", or "unet"
                - "unet*|vae*" : anything with base name starting with "unet" OR starting with "vae"
            collection: Optional collection to filter by
            load_id: Optional load_id to filter by
            return_dict_with_names:
                                    If True, returns a dictionary with component names as keys, throw an error if
                                    multiple components with the same name are found If False, returns a dictionary
                                    with component IDs as keys
