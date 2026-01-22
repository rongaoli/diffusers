from typing import Union

from ..utils import is_torch_available, logging

if is_torch_available():
    from .adaptive_projected_guidance import AdaptiveProjectedGuidance
    from .adaptive_projected_guidance_mix import AdaptiveProjectedMixGuidance
    from .auto_guidance import AutoGuidance
    from .classifier_free_guidance import ClassifierFreeGuidance
    from .classifier_free_zero_star_guidance import ClassifierFreeZeroStarGuidance
    from .frequency_decoupled_guidance import FrequencyDecoupledGuidance
    from .guider_utils import BaseGuidance
    from .magnitude_aware_guidance import MagnitudeAwareGuidance
    from .perturbed_attention_guidance import PerturbedAttentionGuidance
    from .skip_layer_guidance import SkipLayerGuidance
    from .smoothed_energy_guidance import SmoothedEnergyGuidance
    from .tangential_classifier_free_guidance import TangentialClassifierFreeGuidance
