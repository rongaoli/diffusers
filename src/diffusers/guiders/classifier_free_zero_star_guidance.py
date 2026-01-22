import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from ..configuration_utils import register_to_config
from .guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg

if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState

class ClassifierFreeZeroStarGuidance(BaseGuidance):
    
    class ClassifierFreeZeroStarGuidance(BaseGuidance):


    _input_predictions = ["pred_cond", "pred_uncond"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 7.5,
        zero_init_steps: int = 1,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)

        self.guidance_scale = guidance_scale
        self.zero_init_steps = zero_init_steps
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

    def prepare_inputs(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> List["BlockState"]:
        tuple_indices = [0] if self.num_conditions == 1 else [0, 1]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, self._input_predictions):
            data_batch = self._prepare_batch(data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def prepare_inputs_from_block_state(
        self, data: "BlockState", input_fields: Dict[str, Union[str, Tuple[str, str]]]
    ) -> List["BlockState"]:
        tuple_indices = [0] if self.num_conditions == 1 else [0, 1]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, self._input_predictions):
            data_batch = self._prepare_batch_from_block_state(input_fields, data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def forward(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> GuiderOutput:
        pred = None

        # YiYi Notes: add default behavior for self._enabled == False
        if not self._enabled:
            pred = pred_cond

        elif self._step < self.zero_init_steps:
            pred = torch.zeros_like(pred_cond)
        elif not self._is_cfg_enabled():
            pred = pred_cond
        else:
            pred_cond_flat = pred_cond.flatten(1)
            pred_uncond_flat = pred_uncond.flatten(1)
            alpha = cfg_zero_star_scale(pred_cond_flat, pred_uncond_flat)
            alpha = alpha.view(-1, *(1,) * (len(pred_cond.shape) - 1))
            pred_uncond = pred_uncond * alpha
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)

    @property
    def is_conditional(self) -> bool:
        return self._count_prepared == 1

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_cfg_enabled():
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

def cfg_zero_star_scale(cond: torch.Tensor, uncond: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cond_dtype = cond.dtype
    cond = cond.float()
    uncond = uncond.float()
    dot_product = torch.sum(cond * uncond, dim=1, keepdim=True)
    squared_norm = torch.sum(uncond**2, dim=1, keepdim=True) + eps
    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    scale = dot_product / squared_norm
    return scale.to(dtype=cond_dtype)
