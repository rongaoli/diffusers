# coding=utf-8
import contextlib
import os
import tempfile
from typing import TYPE_CHECKING, Dict

from huggingface_hub import DDUFEntry
from tqdm import tqdm

from ..utils import is_safetensors_available, is_transformers_available, is_transformers_version

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

if is_transformers_available():
    from transformers import PreTrainedModel, PreTrainedTokenizer

if is_safetensors_available():
    import safetensors.torch

def _load_tokenizer_from_dduf(
    cls: "PreTrainedTokenizer", name: str, dduf_entries: Dict[str, DDUFEntry], **kwargs
) -> "PreTrainedTokenizer":

    with tempfile.TemporaryDirectory() as tmp_dir:
        for entry_name, entry in dduf_entries.items():
            if entry_name.startswith(name + "/"):
                tmp_entry_path = os.path.join(tmp_dir, *entry_name.split("/"))
                # need to create intermediary directory if they don't exist
                os.makedirs(os.path.dirname(tmp_entry_path), exist_ok=True)
                with open(tmp_entry_path, "wb") as f:
                    with entry.as_mmap() as mm:
                        f.write(mm)
        return cls.from_pretrained(os.path.dirname(tmp_entry_path), **kwargs)

def _load_transformers_model_from_dduf(
    cls: "PreTrainedModel", name: str, dduf_entries: Dict[str, DDUFEntry], **kwargs
) -> "PreTrainedModel":

    config_file = dduf_entries.get(f"{name}/config.json")
    if config_file is None:
        raise EnvironmentError(
            f"Could not find a config.json file for component {name} in DDUF file (contains {dduf_entries.keys()})."
        )
    generation_config = dduf_entries.get(f"{name}/generation_config.json", None)

    weight_files = [
        entry
        for entry_name, entry in dduf_entries.items()
        if entry_name.startswith(f"{name}/") and entry_name.endswith(".safetensors")
    ]
    if not weight_files:
        raise EnvironmentError(
            f"Could not find any weight file for component {name} in DDUF file (contains {dduf_entries.keys()})."
        )
    if not is_safetensors_available():
        raise EnvironmentError(
            "Safetensors is not available, cannot load model from DDUF. Please `pip install safetensors`."
        )
    if is_transformers_version("<", "4.47.0"):
        raise ImportError(
            "You need to install `transformers>4.47.0` in order to load a transformers model from a DDUF file. "
            "You can install it with: `pip install --upgrade transformers`"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        from transformers import AutoConfig, GenerationConfig

        tmp_config_file = os.path.join(tmp_dir, "config.json")
        with open(tmp_config_file, "w") as f:
            f.write(config_file.read_text())
        config = AutoConfig.from_pretrained(tmp_config_file)
        if generation_config is not None:
            tmp_generation_config_file = os.path.join(tmp_dir, "generation_config.json")
            with open(tmp_generation_config_file, "w") as f:
                f.write(generation_config.read_text())
            generation_config = GenerationConfig.from_pretrained(tmp_generation_config_file)
        state_dict = {}
        with contextlib.ExitStack() as stack:
            for entry in tqdm(weight_files, desc="Loading state_dict"):  # Loop over safetensors files
                # Memory-map the safetensors file
                mmap = stack.enter_context(entry.as_mmap())
                # Load tensors from the memory-mapped file
                tensors = safetensors.torch.load(mmap)
                # Update the state dictionary with tensors
                state_dict.update(tensors)
            return cls.from_pretrained(
                pretrained_model_name_or_path=None,
                config=config,
                generation_config=generation_config,
                state_dict=state_dict,
                **kwargs,
            )
