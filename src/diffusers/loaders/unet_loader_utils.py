import copy
from typing import TYPE_CHECKING, Dict, List, Union

from torch import nn

from ..utils import logging

if TYPE_CHECKING:
    # import here to avoid circular imports
    from ..models import UNet2DConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def _translate_into_actual_layer_name(name):

    if name == "mid":
        return "mid_block.attentions.0"

    updown, block, attn = name.split(".")

    updown = updown.replace("down", "down_blocks").replace("up", "up_blocks")
    block = block.replace("block_", "")
    attn = "attentions." + attn

    return ".".join((updown, block, attn))

def _maybe_expand_lora_scales(
    unet: "UNet2DConditionModel", weight_scales: List[Union[float, Dict]], default_scale=1.0
):
    blocks_with_transformer = {
        "down": [i for i, block in enumerate(unet.down_blocks) if hasattr(block, "attentions")],
        "up": [i for i, block in enumerate(unet.up_blocks) if hasattr(block, "attentions")],
    }
    transformer_per_block = {"down": unet.config.layers_per_block, "up": unet.config.layers_per_block + 1}

    expanded_weight_scales = [
        _maybe_expand_lora_scales_for_one_adapter(
            weight_for_adapter,
            blocks_with_transformer,
            transformer_per_block,
            model=unet,
            default_scale=default_scale,
        )
        for weight_for_adapter in weight_scales
    ]

    return expanded_weight_scales

def _maybe_expand_lora_scales_for_one_adapter(
    scales: Union[float, Dict],
    blocks_with_transformer: Dict[str, int],
    transformer_per_block: Dict[str, int],
    model: nn.Module,
    default_scale: float = 1.0,
):

    if sorted(blocks_with_transformer.keys()) != ["down", "up"]:
        raise ValueError("blocks_with_transformer needs to be a dict with keys `'down' and `'up'`")

    if sorted(transformer_per_block.keys()) != ["down", "up"]:
        raise ValueError("transformer_per_block needs to be a dict with keys `'down' and `'up'`")

    if not isinstance(scales, dict):
        # don't expand if scales is a single number
        return scales

    scales = copy.deepcopy(scales)

    if "mid" not in scales:
        scales["mid"] = default_scale
    elif isinstance(scales["mid"], list):
        if len(scales["mid"]) == 1:
            scales["mid"] = scales["mid"][0]
        else:
            raise ValueError(f"Expected 1 scales for mid, got {len(scales['mid'])}.")

    for updown in ["up", "down"]:
        if updown not in scales:
            scales[updown] = default_scale

        # eg {"down": 1} to {"down": {"block_1": 1, "block_2": 1}}}
        if not isinstance(scales[updown], dict):
            scales[updown] = {f"block_{i}": copy.deepcopy(scales[updown]) for i in blocks_with_transformer[updown]}

        # eg {"down": {"block_1": 1}} to {"down": {"block_1": [1, 1]}}
        for i in blocks_with_transformer[updown]:
            block = f"block_{i}"
            # set not assigned blocks to default scale
            if block not in scales[updown]:
                scales[updown][block] = default_scale
            if not isinstance(scales[updown][block], list):
                scales[updown][block] = [scales[updown][block] for _ in range(transformer_per_block[updown])]
            elif len(scales[updown][block]) == 1:
                # a list specifying scale to each masked IP input
                scales[updown][block] = scales[updown][block] * transformer_per_block[updown]
            elif len(scales[updown][block]) != transformer_per_block[updown]:
                raise ValueError(
                    f"Expected {transformer_per_block[updown]} scales for {updown}.{block}, got {len(scales[updown][block])}."
                )

        # eg {"down": "block_1": [1, 1]}}  to {"down.block_1.0": 1, "down.block_1.1": 1}
        for i in blocks_with_transformer[updown]:
            block = f"block_{i}"
            for tf_idx, value in enumerate(scales[updown][block]):
                scales[f"{updown}.{block}.{tf_idx}"] = value

        del scales[updown]

    state_dict = model.state_dict()
    for layer in scales.keys():
        if not any(_translate_into_actual_layer_name(layer) in module for module in state_dict.keys()):
            raise ValueError(
                f"Can't set lora scale for layer {layer}. It either doesn't exist in this unet or it has no attentions."
            )

    return {_translate_into_actual_layer_name(name): weight for name, weight in scales.items()}
