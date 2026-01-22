import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from ..configuration_utils import register_to_config
from .guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg

if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState

class AdaptiveProjectedGuidance(BaseGuidance):
    
    class AdaptiveProjectedGuidance(BaseGuidance):


    _input_predictions = ["pred_cond", "pred_uncond"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 7.5,
        adaptive_projected_guidance_momentum: Optional[float] = None,
        adaptive_projected_guidance_rescale: float = 15.0,
        eta: float = 1.0,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)

        self.guidance_scale = guidance_scale
        self.adaptive_projected_guidance_momentum = adaptive_projected_guidance_momentum
        self.adaptive_projected_guidance_rescale = adaptive_projected_guidance_rescale
        self.eta = eta
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation
        self.momentum_buffer = None

    def prepare_inputs(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> List["BlockState"]:
        if self._step == 0:
            if self.adaptive_projected_guidance_momentum is not None:
                self.momentum_buffer = MomentumBuffer(self.adaptive_projected_guidance_momentum)
        tuple_indices = [0] if self.num_conditions == 1 else [0, 1]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, self._input_predictions):
            data_batch = self._prepare_batch(data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def prepare_inputs_from_block_state(
        self, data: "BlockState", input_fields: Dict[str, Union[str, Tuple[str, str]]]
    ) -> List["BlockState"]:
        if self._step == 0:
            if self.adaptive_projected_guidance_momentum is not None:
                self.momentum_buffer = MomentumBuffer(self.adaptive_projected_guidance_momentum)
        tuple_indices = [0] if self.num_conditions == 1 else [0, 1]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, self._input_predictions):
            data_batch = self._prepare_batch_from_block_state(input_fields, data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def forward(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> GuiderOutput:
        pred = None

        if not self._is_apg_enabled():
            pred = pred_cond
        else:
            pred = normalized_guidance(
                pred_cond,
                pred_uncond,
                self.guidance_scale,
                self.momentum_buffer,
                self.eta,
                self.adaptive_projected_guidance_rescale,
                self.use_original_formulation,
            )

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)

    @property
    def is_conditional(self) -> bool:
        return self._count_prepared == 1

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_apg_enabled():
            num_conditions += 1
        return num_conditions

    def _is_apg_enabled(self) -> bool:
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

class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

    def __repr__(self) -> str:
        class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

    def __repr__(self) -> str:
        
        class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

    def __repr__(self) -> str:
        class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

    def __repr__(self) -> str:

        if isinstance(self.running_average, torch.Tensor):
            shape = tuple(self.running_average.shape)

            # Calculate statistics
            with torch.no_grad():
                stats = {
                    "mean": self.running_average.mean().item(),
                    "std": self.running_average.std().item(),
                    "min": self.running_average.min().item(),
                    "max": self.running_average.max().item(),
                }

            # Get a slice (max 3 elements per dimension)
            slice_indices = tuple(slice(None, min(3, dim)) for dim in shape)
            sliced_data = self.running_average[slice_indices]

            # Format the slice for display (conver...
            slice_str = str(sliced_data.detach().float().cpu().numpy())
            if len(slice_str) > 200:  # Truncate if too long
                slice_str = slice_str[:200] + "..."

            stats_str = ", ".join([f"{k}={v:.4f}" for k, v in stats.items()])

            return (
                f"MomentumBuffer(\n"
                f"  momentum={self.momentum},\n"
                f"  shape={shape},\n"
                f"  stats=[{stats_str}],\n"
                f"  slice={slice_str}\n"
                f")"
            )
        else:
            return f"MomentumBuffer(momentum={self.momentum}, running_average={self.running_average})"

def normalized_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: Optional[MomentumBuffer] = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
    use_original_formulation: bool = False,
):
    diff = pred_cond - pred_uncond
    dim = [-i for i in range(1, len(diff.shape))]

    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dim, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    v0, v1 = diff.double(), pred_cond.double()
    v1 = torch.nn.functional.normalize(v1, dim=dim)
    v0_parallel = (v0 * v1).sum(dim=dim, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    diff_parallel, diff_orthogonal = v0_parallel.type_as(diff), v0_orthogonal.type_as(diff)
    normalized_update = diff_orthogonal + eta * diff_parallel

    pred = pred_cond if use_original_formulation else pred_uncond
    pred = pred + guidance_scale * normalized_update

    return pred
