import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from ..configuration_utils import register_to_config
from ..hooks import HookRegistry
from ..hooks.smoothed_energy_guidance_utils import SmoothedEnergyGuidanceConfig, _apply_smoothed_energy_guidance_hook
from .guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg

if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState

class SmoothedEnergyGuidance(BaseGuidance):
    
    class SmoothedEnergyGuidance(BaseGuidance):


    _input_predictions = ["pred_cond", "pred_uncond", "pred_cond_seg"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 7.5,
        seg_guidance_scale: float = 2.8,
        seg_blur_sigma: float = 9999999.0,
        seg_blur_threshold_inf: float = 9999.0,
        seg_guidance_start: float = 0.0,
        seg_guidance_stop: float = 1.0,
        seg_guidance_layers: Optional[int] = None,
        seg_guidance_config: Union[SmoothedEnergyGuidanceConfig, List[SmoothedEnergyGuidanceConfig]] = None,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)

        self.guidance_scale = guidance_scale
        self.seg_guidance_scale = seg_guidance_scale
        self.seg_blur_sigma = seg_blur_sigma
        self.seg_blur_threshold_inf = seg_blur_threshold_inf
        self.seg_guidance_start = seg_guidance_start
        self.seg_guidance_stop = seg_guidance_stop
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

        if not (0.0 <= seg_guidance_start < 1.0):
            raise ValueError(f"Expected `seg_guidance_start` to be between 0.0 and 1.0, but got {seg_guidance_start}.")
        if not (seg_guidance_start <= seg_guidance_stop <= 1.0):
            raise ValueError(f"Expected `seg_guidance_stop` to be between 0.0 and 1.0, but got {seg_guidance_stop}.")

        if seg_guidance_layers is None and seg_guidance_config is None:
            raise ValueError(
                "Either `seg_guidance_layers` or `seg_guidance_config` must be provided to enable Smoothed Energy Guidance."
            )
        if seg_guidance_layers is not None and seg_guidance_config is not None:
            raise ValueError("Only one of `seg_guidance_layers` or `seg_guidance_config` can be provided.")

        if seg_guidance_layers is not None:
            if isinstance(seg_guidance_layers, int):
                seg_guidance_layers = [seg_guidance_layers]
            if not isinstance(seg_guidance_layers, list):
                raise ValueError(
                    f"Expected `seg_guidance_layers` to be an int or a list of ints, but got {type(seg_guidance_layers)}."
                )
            seg_guidance_config = [SmoothedEnergyGuidanceConfig(layer, fqn="auto") for layer in seg_guidance_layers]

        if isinstance(seg_guidance_config, dict):
            seg_guidance_config = SmoothedEnergyGuidanceConfig.from_dict(seg_guidance_config)

        if isinstance(seg_guidance_config, SmoothedEnergyGuidanceConfig):
            seg_guidance_config = [seg_guidance_config]

        if not isinstance(seg_guidance_config, list):
            raise ValueError(
                f"Expected `seg_guidance_config` to be a SmoothedEnergyGuidanceConfig or a list of SmoothedEnergyGuidanceConfig, but got {type(seg_guidance_config)}."
            )
        elif isinstance(next(iter(seg_guidance_config), None), dict):
            seg_guidance_config = [SmoothedEnergyGuidanceConfig.from_dict(config) for config in seg_guidance_config]

        self.seg_guidance_config = seg_guidance_config
        self._seg_layer_hook_names = [f"SmoothedEnergyGuidance_{i}" for i in range(len(self.seg_guidance_config))]

    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        if self._is_seg_enabled() and self.is_conditional and self._count_prepared > 1:
            for name, config in zip(self._seg_layer_hook_names, self.seg_guidance_config):
                _apply_smoothed_energy_guidance_hook(denoiser, config, self.seg_blur_sigma, name=name)

    def cleanup_models(self, denoiser: torch.nn.Module):
        if self._is_seg_enabled() and self.is_conditional and self._count_prepared > 1:
            registry = HookRegistry.check_if_exists_or_initialize(denoiser)
            # Remove the hooks after inference
            for hook_name in self._seg_layer_hook_names:
                registry.remove_hook(hook_name, recurse=True)

    def prepare_inputs(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> List["BlockState"]:
        if self.num_conditions == 1:
            tuple_indices = [0]
            input_predictions = ["pred_cond"]
        elif self.num_conditions == 2:
            tuple_indices = [0, 1]
            input_predictions = (
                ["pred_cond", "pred_uncond"] if self._is_cfg_enabled() else ["pred_cond", "pred_cond_seg"]
            )
        else:
            tuple_indices = [0, 1, 0]
            input_predictions = ["pred_cond", "pred_uncond", "pred_cond_seg"]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, input_predictions):
            data_batch = self._prepare_batch(data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def prepare_inputs_from_block_state(
        self, data: "BlockState", input_fields: Dict[str, Union[str, Tuple[str, str]]]
    ) -> List["BlockState"]:
        if self.num_conditions == 1:
            tuple_indices = [0]
            input_predictions = ["pred_cond"]
        elif self.num_conditions == 2:
            tuple_indices = [0, 1]
            input_predictions = (
                ["pred_cond", "pred_uncond"] if self._is_cfg_enabled() else ["pred_cond", "pred_cond_seg"]
            )
        else:
            tuple_indices = [0, 1, 0]
            input_predictions = ["pred_cond", "pred_uncond", "pred_cond_seg"]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, input_predictions):
            data_batch = self._prepare_batch_from_block_state(input_fields, data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def forward(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: Optional[torch.Tensor] = None,
        pred_cond_seg: Optional[torch.Tensor] = None,
    ) -> GuiderOutput:
        pred = None

        if not self._is_cfg_enabled() and not self._is_seg_enabled():
            pred = pred_cond
        elif not self._is_cfg_enabled():
            shift = pred_cond - pred_cond_seg
            pred = pred_cond if self.use_original_formulation else pred_cond_seg
            pred = pred + self.seg_guidance_scale * shift
        elif not self._is_seg_enabled():
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift
        else:
            shift = pred_cond - pred_uncond
            shift_seg = pred_cond - pred_cond_seg
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift + self.seg_guidance_scale * shift_seg

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)

    @property
    def is_conditional(self) -> bool:
        return self._count_prepared == 1 or self._count_prepared == 3

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_cfg_enabled():
            num_conditions += 1
        if self._is_seg_enabled():
            num_conditions += 1
        return num_conditions

    def _is_cfg_enabled(self) -> bool:
        if not self._enabled:
            return False

        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self._start * self._num_inference_steps)
            skip_stop_step = int(self._stop * self._num_inference_steps)
            is_within_range = skip_start_step <= self._step < skip_stop_step

        is_close = False
        if self.use_original_formulation:
            is_close = math.isclose(self.guidance_scale, 0.0)
        else:
            is_close = math.isclose(self.guidance_scale, 1.0)

        return is_within_range and not is_close

    def _is_seg_enabled(self) -> bool:
        if not self._enabled:
            return False

        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self.seg_guidance_start * self._num_inference_steps)
            skip_stop_step = int(self.seg_guidance_stop * self._num_inference_steps)
            is_within_range = skip_start_step < self._step < skip_stop_step

        is_zero = math.isclose(self.seg_guidance_scale, 0.0)

        return is_within_range and not is_zero
