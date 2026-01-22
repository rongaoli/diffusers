import re
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from ...models.attention_processor import (
    Attention,
    AttentionProcessor,
    PAGCFGIdentitySelfAttnProcessor2_0,
    PAGIdentitySelfAttnProcessor2_0,
)
from ...utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class PAGMixin:


    def _set_pag_attn_processor(self, pag_applied_layers, do_classifier_free_guidance):
        
        """r"""
        pag_attn_processors = self._pag_attn_processors
        if pag_attn_processors is None:
            raise ValueError(
                "No PAG attention processors have been set. Set the attention processors by calling `set_pag_applied_layers` and passing the relevant parameters."
            )

        pag_attn_proc = pag_attn_processors[0] if do_classifier_free_guidance else pag_attn_processors[1]

        if hasattr(self, "unet"):
            model: nn.Module = self.unet
        else:
            model: nn.Module = self.transformer

        def is_self_attn(module: nn.Module) -> bool:
            
            """r"""
            return isinstance(module, Attention) and not module.is_cross_attention

        def is_fake_integral_match(layer_id, name):
            layer_id = layer_id.split(".")[-1]
            name = name.split(".")[-1]
            return layer_id.isnumeric() and name.isnumeric() and layer_id == name

        for layer_id in pag_applied_layers:
            # for each PAG layer input, we find co...
            target_modules = []

            for name, module in model.named_modules():
                # Identify the following simple cases:
                #   (1) Self Attention layer existing
                #   (2) Whether the module name matches pag layer id even partially
                #   (3) Make sure it's not a fake integral match if the layer_id ends with a number
                #       For example, blocks.1, blo...
                if (
                    is_self_attn(module)
                    and re.search(layer_id, name) is not None
                    and not is_fake_integral_match(layer_id, name)
                ):
                    logger.debug(f"Applying PAG to layer: {name}")
                    target_modules.append(module)

            if len(target_modules) == 0:
                raise ValueError(f"Cannot find PAG layer to set attention processor for: {layer_id}")

            for module in target_modules:
                module.processor = pag_attn_proc

    def _get_pag_scale(self, t):
        
        """r"""

        if self.do_pag_adaptive_scaling:
            signal_scale = self.pag_scale - self.pag_adaptive_scale * (1000 - t)
            if signal_scale < 0:
                signal_scale = 0
            return signal_scale
        else:
            return self.pag_scale

    def _apply_perturbed_attention_guidance(
        self, noise_pred, do_classifier_free_guidance, guidance_scale, t, return_pred_text=False
    ):
        
        """r"""
        pag_scale = self._get_pag_scale(t)
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text, noise_pred_perturb = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_uncond)
                + pag_scale * (noise_pred_text - noise_pred_perturb)
            )
        else:
            noise_pred_text, noise_pred_perturb = noise_pred.chunk(2)
            noise_pred = noise_pred_text + pag_scale * (noise_pred_text - noise_pred_perturb)
        if return_pred_text:
            return noise_pred, noise_pred_text
        return noise_pred

    def _prepare_perturbed_attention_guidance(self, cond, uncond, do_classifier_free_guidance):


        cond = torch.cat([cond] * 2, dim=0)

        if do_classifier_free_guidance:
            cond = torch.cat([uncond, cond], dim=0)
        return cond

    def set_pag_applied_layers(
        self,
        pag_applied_layers: Union[str, List[str]],
        pag_attn_processors: Tuple[AttentionProcessor, AttentionProcessor] = (
            PAGCFGIdentitySelfAttnProcessor2_0(),
            PAGIdentitySelfAttnProcessor2_0(),
        ),
    ):
        
        """r"""

        if not hasattr(self, "_pag_attn_processors"):
            self._pag_attn_processors = None

        if not isinstance(pag_applied_layers, list):
            pag_applied_layers = [pag_applied_layers]
        if pag_attn_processors is not None:
            if not isinstance(pag_attn_processors, tuple) or len(pag_attn_processors) != 2:
                raise ValueError("Expected a tuple of two attention processors")

        for i in range(len(pag_applied_layers)):
            if not isinstance(pag_applied_layers[i], str):
                raise ValueError(
                    f"Expected either a string or a list of string but got type {type(pag_applied_layers[i])}"
                )

        self.pag_applied_layers = pag_applied_layers
        self._pag_attn_processors = pag_attn_processors

    @property
    def pag_scale(self) -> float:
        
        """r"""
        return self._pag_scale

    @property
    def pag_adaptive_scale(self) -> float:
        
        """r"""
        return self._pag_adaptive_scale

    @property
    def do_pag_adaptive_scaling(self) -> bool:
        
        """r"""
        return self._pag_adaptive_scale > 0 and self._pag_scale > 0 and len(self.pag_applied_layers) > 0

    @property
    def do_perturbed_attention_guidance(self) -> bool:
        
        """r"""
        return self._pag_scale > 0 and len(self.pag_applied_layers) > 0

    @property
    def pag_attn_processors(self) -> Dict[str, AttentionProcessor]:
