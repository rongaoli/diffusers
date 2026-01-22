State dict utilities: utility methods for converting state dicts easily

import enum
import json

from .import_utils import is_torch_available
from .logging import get_logger

if is_torch_available():
    import torch

logger = get_logger(__name__)

class StateDictType(enum.Enum):


    DIFFUSERS_OLD = "diffusers_old"
    KOHYA_SS = "kohya_ss"
    PEFT = "peft"
    DIFFUSERS = "diffusers"

# We need to define a proper mapping for Unet since it uses different output keys than text encoder
# e.g. to_q_lora -> q_proj / to_q
UNET_TO_DIFFUSERS = {
    ".to_out_lora.up": ".to_out.0.lora_B",
    ".to_out_lora.down": ".to_out.0.lora_A",
    ".to_q_lora.down": ".to_q.lora_A",
    ".to_q_lora.up": ".to_q.lora_B",
    ".to_k_lora.down": ".to_k.lora_A",
    ".to_k_lora.up": ".to_k.lora_B",
    ".to_v_lora.down": ".to_v.lora_A",
    ".to_v_lora.up": ".to_v.lora_B",
    ".lora.up": ".lora_B",
    ".lora.down": ".lora_A",
    ".to_out.lora_magnitude_vector": ".to_out.0.lora_magnitude_vector",
}

CONTROL_LORA_TO_DIFFUSERS = {
    ".to_q.down": ".to_q.lora_A.weight",
    ".to_q.up": ".to_q.lora_B.weight",
    ".to_k.down": ".to_k.lora_A.weight",
    ".to_k.up": ".to_k.lora_B.weight",
    ".to_v.down": ".to_v.lora_A.weight",
    ".to_v.up": ".to_v.lora_B.weight",
    ".to_out.0.down": ".to_out.0.lora_A.weight",
    ".to_out.0.up": ".to_out.0.lora_B.weight",
    ".ff.net.0.proj.down": ".ff.net.0.proj.lora_A.weight",
    ".ff.net.0.proj.up": ".ff.net.0.proj.lora_B.weight",
    ".ff.net.2.down": ".ff.net.2.lora_A.weight",
    ".ff.net.2.up": ".ff.net.2.lora_B.weight",
    ".proj_in.down": ".proj_in.lora_A.weight",
    ".proj_in.up": ".proj_in.lora_B.weight",
    ".proj_out.down": ".proj_out.lora_A.weight",
    ".proj_out.up": ".proj_out.lora_B.weight",
    ".conv.down": ".conv.lora_A.weight",
    ".conv.up": ".conv.lora_B.weight",
    **{f".conv{i}.down": f".conv{i}.lora_A.weight" for i in range(1, 3)},
    **{f".conv{i}.up": f".conv{i}.lora_B.weight" for i in range(1, 3)},
    "conv_in.down": "conv_in.lora_A.weight",
    "conv_in.up": "conv_in.lora_B.weight",
    ".conv_shortcut.down": ".conv_shortcut.lora_A.weight",
    ".conv_shortcut.up": ".conv_shortcut.lora_B.weight",
    **{f".linear_{i}.down": f".linear_{i}.lora_A.weight" for i in range(1, 3)},
    **{f".linear_{i}.up": f".linear_{i}.lora_B.weight" for i in range(1, 3)},
    "time_emb_proj.down": "time_emb_proj.lora_A.weight",
    "time_emb_proj.up": "time_emb_proj.lora_B.weight",
}

DIFFUSERS_TO_PEFT = {
    ".q_proj.lora_linear_layer.up": ".q_proj.lora_B",
    ".q_proj.lora_linear_layer.down": ".q_proj.lora_A",
    ".k_proj.lora_linear_layer.up": ".k_proj.lora_B",
    ".k_proj.lora_linear_layer.down": ".k_proj.lora_A",
    ".v_proj.lora_linear_layer.up": ".v_proj.lora_B",
    ".v_proj.lora_linear_layer.down": ".v_proj.lora_A",
    ".out_proj.lora_linear_layer.up": ".out_proj.lora_B",
    ".out_proj.lora_linear_layer.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
    "text_projection.lora.down.weight": "text_projection.lora_A.weight",
    "text_projection.lora.up.weight": "text_projection.lora_B.weight",
}

DIFFUSERS_OLD_TO_PEFT = {
    ".to_q_lora.up": ".q_proj.lora_B",
    ".to_q_lora.down": ".q_proj.lora_A",
    ".to_k_lora.up": ".k_proj.lora_B",
    ".to_k_lora.down": ".k_proj.lora_A",
    ".to_v_lora.up": ".v_proj.lora_B",
    ".to_v_lora.down": ".v_proj.lora_A",
    ".to_out_lora.up": ".out_proj.lora_B",
    ".to_out_lora.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
}

PEFT_TO_DIFFUSERS = {
    ".q_proj.lora_B": ".q_proj.lora_linear_layer.up",
    ".q_proj.lora_A": ".q_proj.lora_linear_layer.down",
    ".k_proj.lora_B": ".k_proj.lora_linear_layer.up",
    ".k_proj.lora_A": ".k_proj.lora_linear_layer.down",
    ".v_proj.lora_B": ".v_proj.lora_linear_layer.up",
    ".v_proj.lora_A": ".v_proj.lora_linear_layer.down",
    ".out_proj.lora_B": ".out_proj.lora_linear_layer.up",
    ".out_proj.lora_A": ".out_proj.lora_linear_layer.down",
    "to_k.lora_A": "to_k.lora.down",
    "to_k.lora_B": "to_k.lora.up",
    "to_q.lora_A": "to_q.lora.down",
    "to_q.lora_B": "to_q.lora.up",
    "to_v.lora_A": "to_v.lora.down",
    "to_v.lora_B": "to_v.lora.up",
    "to_out.0.lora_A": "to_out.0.lora.down",
    "to_out.0.lora_B": "to_out.0.lora.up",
}

DIFFUSERS_OLD_TO_DIFFUSERS = {
    ".to_q_lora.up": ".q_proj.lora_linear_layer.up",
    ".to_q_lora.down": ".q_proj.lora_linear_layer.down",
    ".to_k_lora.up": ".k_proj.lora_linear_layer.up",
    ".to_k_lora.down": ".k_proj.lora_linear_layer.down",
    ".to_v_lora.up": ".v_proj.lora_linear_layer.up",
    ".to_v_lora.down": ".v_proj.lora_linear_layer.down",
    ".to_out_lora.up": ".out_proj.lora_linear_layer.up",
    ".to_out_lora.down": ".out_proj.lora_linear_layer.down",
    ".to_k.lora_magnitude_vector": ".k_proj.lora_magnitude_vector",
    ".to_v.lora_magnitude_vector": ".v_proj.lora_magnitude_vector",
    ".to_q.lora_magnitude_vector": ".q_proj.lora_magnitude_vector",
    ".to_out.lora_magnitude_vector": ".out_proj.lora_magnitude_vector",
}

PEFT_TO_KOHYA_SS = {
    "lora_A": "lora_down",
    "lora_B": "lora_up",
    # This is not a comprehensive dict as kohya format requires replacing `.` with `_` in keys,
    # adding prefixes and adding alpha values
    # Check `convert_state_dict_to_kohya` for more
}

PEFT_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_PEFT,
    StateDictType.DIFFUSERS: DIFFUSERS_TO_PEFT,
}

DIFFUSERS_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_DIFFUSERS,
    StateDictType.PEFT: PEFT_TO_DIFFUSERS,
}

KOHYA_STATE_DICT_MAPPINGS = {StateDictType.PEFT: PEFT_TO_KOHYA_SS}

KEYS_TO_ALWAYS_REPLACE = {
    ".processor.": ".",
}

def convert_state_dict(state_dict, mapping):

    Simply iterates over the state dict and replaces the patterns in `mapping` with the corresponding values.
                - key: the pattern to replace
                - value: the pattern to replace with
