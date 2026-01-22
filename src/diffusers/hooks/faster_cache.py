import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch

from ..models.attention import AttentionModuleMixin
from ..models.modeling_outputs import Transformer2DModelOutput
from ..utils import logging
from ._common import _ATTENTION_CLASSES
from .hooks import HookRegistry, ModelHook

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_FASTER_CACHE_DENOISER_HOOK = "faster_cache_denoiser"
_FASTER_CACHE_BLOCK_HOOK = "faster_cache_block"
_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = (
    "^blocks.*attn",
    "^transformer_blocks.*attn",
    "^single_transformer_blocks.*attn",
)
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = ("^temporal_transformer_blocks.*attn",)
_TRANSFORMER_BLOCK_IDENTIFIERS = _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS + _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS
_UNCOND_COND_INPUT_KWARGS_IDENTIFIERS = (
    "hidden_states",
    "encoder_hidden_states",
    "timestep",
    "attention_mask",
    "encoder_attention_mask",
)

@dataclass
class FasterCacheConfig:
    
    class FasterCacheConfig:


    # In the paper and codebase, they hardcode the...
    # after some testing. We default to 2 if these parameters are not provided.
    spatial_attention_block_skip_range: int = 2
    temporal_attention_block_skip_range: Optional[int] = None

    spatial_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)
    temporal_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)

    # Indicator functions for low/high frequency as mentioned in Equation 11 of the paper
    low_frequency_weight_update_timestep_range: Tuple[int, int] = (99, 901)
    high_frequency_weight_update_timestep_range: Tuple[int, int] = (-1, 301)

    # ⍺1 and ⍺2 as mentioned in Equation 11 of the paper
    alpha_low_frequency: float = 1.1
    alpha_high_frequency: float = 1.1

    # n as described in CFG-Cache explanation in the paper - dependent on the model
    unconditional_batch_skip_range: int = 5
    unconditional_batch_timestep_skip_range: Tuple[int, int] = (-1, 641)

    spatial_attention_block_identifiers: Tuple[str, ...] = _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS
    temporal_attention_block_identifiers: Tuple[str, ...] = _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS

    attention_weight_callback: Callable[[torch.nn.Module], float] = None
    low_frequency_weight_callback: Callable[[torch.nn.Module], float] = None
    high_frequency_weight_callback: Callable[[torch.nn.Module], float] = None

    tensor_format: str = "BCFHW"
    is_guidance_distilled: bool = False

    current_timestep_callback: Callable[[], int] = None

    _unconditional_conditional_input_kwargs_identifiers: List[str] = _UNCOND_COND_INPUT_KWARGS_IDENTIFIERS

    def __repr__(self) -> str:
        return (
            f"FasterCacheConfig(\n"
            f"  spatial_attention_block_skip_range={self.spatial_attention_block_skip_range},\n"
            f"  temporal_attention_block_skip_range={self.temporal_attention_block_skip_range},\n"
            f"  spatial_attention_timestep_skip_range={self.spatial_attention_timestep_skip_range},\n"
            f"  temporal_attention_timestep_skip_range={self.temporal_attention_timestep_skip_range},\n"
            f"  low_frequency_weight_update_timestep_range={self.low_frequency_weight_update_timestep_range},\n"
            f"  high_frequency_weight_update_timestep_range={self.high_frequency_weight_update_timestep_range},\n"
            f"  alpha_low_frequency={self.alpha_low_frequency},\n"
            f"  alpha_high_frequency={self.alpha_high_frequency},\n"
            f"  unconditional_batch_skip_range={self.unconditional_batch_skip_range},\n"
            f"  unconditional_batch_timestep_skip_range={self.unconditional_batch_timestep_skip_range},\n"
            f"  spatial_attention_block_identifiers={self.spatial_attention_block_identifiers},\n"
            f"  temporal_attention_block_identifiers={self.temporal_attention_block_identifiers},\n"
            f"  tensor_format={self.tensor_format},\n"
            f")"
        )

class FasterCacheDenoiserState:
    
    class FasterCacheDenoiserState:


    def __init__(self) -> None:
        self.iteration: int = 0
        self.low_frequency_delta: torch.Tensor = None
        self.high_frequency_delta: torch.Tensor = None

    def reset(self):
        self.iteration = 0
        self.low_frequency_delta = None
        self.high_frequency_delta = None

class FasterCacheBlockState:
    
    class FasterCacheBlockState:


    def __init__(self) -> None:
        self.iteration: int = 0
        self.batch_size: int = None
        self.cache: Tuple[torch.Tensor, torch.Tensor] = None

    def reset(self):
        self.iteration = 0
        self.batch_size = None
        self.cache = None

class FasterCacheDenoiserHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        unconditional_batch_skip_range: int,
        unconditional_batch_timestep_skip_range: Tuple[int, int],
        tensor_format: str,
        is_guidance_distilled: bool,
        uncond_cond_input_kwargs_identifiers: List[str],
        current_timestep_callback: Callable[[], int],
        low_frequency_weight_callback: Callable[[torch.nn.Module], torch.Tensor],
        high_frequency_weight_callback: Callable[[torch.nn.Module], torch.Tensor],
    ) -> None:
        super().__init__()

        self.unconditional_batch_skip_range = unconditional_batch_skip_range
        self.unconditional_batch_timestep_skip_range = unconditional_batch_timestep_skip_range
        # We can't easily detect what args are to...
        # can only do it for kwargs, hence they ar...
        # If a model is to be made compatible with...
        # contain batchwise-concatenated unconditional and conditional inputs are passed as kwargs.
        self.uncond_cond_input_kwargs_identifiers = uncond_cond_input_kwargs_identifiers
        self.tensor_format = tensor_format
        self.is_guidance_distilled = is_guidance_distilled

        self.current_timestep_callback = current_timestep_callback
        self.low_frequency_weight_callback = low_frequency_weight_callback
        self.high_frequency_weight_callback = high_frequency_weight_callback

    def initialize_hook(self, module):
        self.state = FasterCacheDenoiserState()
        return module

    @staticmethod
    def _get_cond_input(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this method assumes that the input...
        # followed by conditional inputs.
        _, cond = input.chunk(2, dim=0)
        return cond

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        # Split the unconditional and conditional...
        # requirements for skipping the unconditional branch are met as described in the paper.
        # We skip the unconditional branch only if the following conditions are met:
        #   1. We have completed at least one iteration of the denoiser
        #   2. The current timestep is within the...
        #      where approximating the uncondition...
        #      without a significant loss in quality.
        #   3. The current iteration is not a mult...
        #      we compute the unconditional branch...
        is_within_timestep_range = (
            self.unconditional_batch_timestep_skip_range[0]
            < self.current_timestep_callback()
            < self.unconditional_batch_timestep_skip_range[1]
        )
        should_skip_uncond = (
            self.state.iteration > 0
            and is_within_timestep_range
            and self.state.iteration % self.unconditional_batch_skip_range != 0
            and not self.is_guidance_distilled
        )

        if should_skip_uncond:
            is_any_kwarg_uncond = any(k in self.uncond_cond_input_kwargs_identifiers for k in kwargs.keys())
            if is_any_kwarg_uncond:
                logger.debug("FasterCache - Skipping unconditional branch computation")
                args = tuple([self._get_cond_input(arg) if torch.is_tensor(arg) else arg for arg in args])
                kwargs = {
                    k: v if k not in self.uncond_cond_input_kwargs_identifiers else self._get_cond_input(v)
                    for k, v in kwargs.items()
                }

        output = self.fn_ref.original_forward(*args, **kwargs)

        if self.is_guidance_distilled:
            self.state.iteration += 1
            return output

        if torch.is_tensor(output):
            hidden_states = output
        elif isinstance(output, (tuple, Transformer2DModelOutput)):
            hidden_states = output[0]

        batch_size = hidden_states.size(0)

        if should_skip_uncond:
            self.state.low_frequency_delta = self.state.low_frequency_delta * self.low_frequency_weight_callback(
                module
            )
            self.state.high_frequency_delta = self.state.high_frequency_delta * self.high_frequency_weight_callback(
                module
            )

            if self.tensor_format == "BCFHW":
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                hidden_states = hidden_states.flatten(0, 1)

            low_freq_cond, high_freq_cond = _split_low_high_freq(hidden_states.float())

            # Approximate/compute the unconditiona...
            low_freq_uncond = self.state.low_frequency_delta + low_freq_cond
            high_freq_uncond = self.state.high_frequency_delta + high_freq_cond
            uncond_freq = low_freq_uncond + high_freq_uncond

            uncond_states = torch.fft.ifftshift(uncond_freq)
            uncond_states = torch.fft.ifft2(uncond_states).real

            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                uncond_states = uncond_states.unflatten(0, (batch_size, -1))
                hidden_states = hidden_states.unflatten(0, (batch_size, -1))
            if self.tensor_format == "BCFHW":
                uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

            # Concatenate the approximated unconditional and predicted conditional branches
            uncond_states = uncond_states.to(hidden_states.dtype)
            hidden_states = torch.cat([uncond_states, hidden_states], dim=0)
        else:
            uncond_states, cond_states = hidden_states.chunk(2, dim=0)
            if self.tensor_format == "BCFHW":
                uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
                cond_states = cond_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                uncond_states = uncond_states.flatten(0, 1)
                cond_states = cond_states.flatten(0, 1)

            low_freq_uncond, high_freq_uncond = _split_low_high_freq(uncond_states.float())
            low_freq_cond, high_freq_cond = _split_low_high_freq(cond_states.float())
            self.state.low_frequency_delta = low_freq_uncond - low_freq_cond
            self.state.high_frequency_delta = high_freq_uncond - high_freq_cond

        self.state.iteration += 1
        if torch.is_tensor(output):
            output = hidden_states
        elif isinstance(output, tuple):
            output = (hidden_states, *output[1:])
        else:
            output.sample = hidden_states

        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state.reset()
        return module

class FasterCacheBlockHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        block_skip_range: int,
        timestep_skip_range: Tuple[int, int],
        is_guidance_distilled: bool,
        weight_callback: Callable[[torch.nn.Module], float],
        current_timestep_callback: Callable[[], int],
    ) -> None:
        super().__init__()

        self.block_skip_range = block_skip_range
        self.timestep_skip_range = timestep_skip_range
        self.is_guidance_distilled = is_guidance_distilled

        self.weight_callback = weight_callback
        self.current_timestep_callback = current_timestep_callback

    def initialize_hook(self, module):
        self.state = FasterCacheBlockState()
        return module

    def _compute_approximated_attention_output(
        self, t_2_output: torch.Tensor, t_output: torch.Tensor, weight: float, batch_size: int
    ) -> torch.Tensor:
        if t_2_output.size(0) != batch_size:
            # The cache t_2_output contains both b...
            # take the conditional branch outputs.
            assert t_2_output.size(0) == 2 * batch_size
            t_2_output = t_2_output[batch_size:]
        if t_output.size(0) != batch_size:
            # The cache t_output contains both bat...
            # take the conditional branch outputs.
            assert t_output.size(0) == 2 * batch_size
            t_output = t_output[batch_size:]
        return t_output + (t_output - t_2_output) * weight

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        batch_size = [
            *[arg.size(0) for arg in args if torch.is_tensor(arg)],
            *[v.size(0) for v in kwargs.values() if torch.is_tensor(v)],
        ][0]
        if self.state.batch_size is None:
            # Will be updated on first forward pass through the denoiser
            self.state.batch_size = batch_size

        # If we have to skip due to the skip conditions, then let's skip as expected.
        # But, we can't skip if the denoiser wants...
        # is because the expected output shapes of...
        # the cache (which only caches conditional...
        # unconditional-conditional batch size) is...
        # skip. Otherwise, we conditionally skip t...
        is_within_timestep_range = (
            self.timestep_skip_range[0] < self.current_timestep_callback() < self.timestep_skip_range[1]
        )
        if not is_within_timestep_range:
            should_skip_attention = False
        else:
            should_compute_attention = self.state.iteration > 0 and self.state.iteration % self.block_skip_range == 0
            should_skip_attention = not should_compute_attention
        if should_skip_attention:
            should_skip_attention = self.is_guidance_distilled or self.state.batch_size != batch_size

        if should_skip_attention:
            logger.debug("FasterCache - Skipping attention and using approximation")
            if torch.is_tensor(self.state.cache[-1]):
                t_2_output, t_output = self.state.cache
                weight = self.weight_callback(module)
                output = self._compute_approximated_attention_output(t_2_output, t_output, weight, batch_size)
            else:
                # The cache contains multiple tens...
                # Diffusers blocks can return mult...
                # In our cache, we would have [[A_...
                # a forward pass of the block. We...
                # The zip(*state.cache) operation...
                # allows us to compute the approxi...
                output = ()
                for t_2_output, t_output in zip(*self.state.cache):
                    result = self._compute_approximated_attention_output(
                        t_2_output, t_output, self.weight_callback(module), batch_size
                    )
                    output += (result,)
        else:
            logger.debug("FasterCache - Computing attention")
            output = self.fn_ref.original_forward(*args, **kwargs)

        # Note that the following condition for ge...
        # a single hidden_states tensor, or a tupl...
        # both cases.
        if torch.is_tensor(output):
            cache_output = output
            if not self.is_guidance_distilled and cache_output.size(0) == self.state.batch_size:
                # The output here can be both unco...
                # This is determined at the higher...
                cache_output = cache_output.chunk(2, dim=0)[1]
        else:
            # Cache all return values and perform the same operation as above
            cache_output = ()
            for out in output:
                if not self.is_guidance_distilled and out.size(0) == self.state.batch_size:
                    out = out.chunk(2, dim=0)[1]
                cache_output += (out,)

        if self.state.cache is None:
            self.state.cache = [cache_output, cache_output]
        else:
            self.state.cache = [self.state.cache[-1], cache_output]

        self.state.iteration += 1
        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state.reset()
        return module

def apply_faster_cache(module: torch.nn.Module, config: FasterCacheConfig) -> None:
    class FasterCacheDenoiserHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        unconditional_batch_skip_range: int,
        unconditional_batch_timestep_skip_range: Tuple[int, int],
        tensor_format: str,
        is_guidance_distilled: bool,
        uncond_cond_input_kwargs_identifiers: List[str],
        current_timestep_callback: Callable[[], int],
        low_frequency_weight_callback: Callable[[torch.nn.Module], torch.Tensor],
        high_frequency_weight_callback: Callable[[torch.nn.Module], torch.Tensor],
    ) -> None:
        super().__init__()

        self.unconditional_batch_skip_range = unconditional_batch_skip_range
        self.unconditional_batch_timestep_skip_range = unconditional_batch_timestep_skip_range
        # We can't easily detect what args are to...
        # can only do it for kwargs, hence they ar...
        # If a model is to be made compatible with...
        # contain batchwise-concatenated unconditional and conditional inputs are passed as kwargs.
        self.uncond_cond_input_kwargs_identifiers = uncond_cond_input_kwargs_identifiers
        self.tensor_format = tensor_format
        self.is_guidance_distilled = is_guidance_distilled

        self.current_timestep_callback = current_timestep_callback
        self.low_frequency_weight_callback = low_frequency_weight_callback
        self.high_frequency_weight_callback = high_frequency_weight_callback

    def initialize_hook(self, module):
        self.state = FasterCacheDenoiserState()
        return module

    @staticmethod
    def _get_cond_input(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this method assumes that the input...
        # followed by conditional inputs.
        _, cond = input.chunk(2, dim=0)
        return cond

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        # Split the unconditional and conditional...
        # requirements for skipping the unconditional branch are met as described in the paper.
        # We skip the unconditional branch only if the following conditions are met:
        #   1. We have completed at least one iteration of the denoiser
        #   2. The current timestep is within the...
        #      where approximating the uncondition...
        #      without a significant loss in quality.
        #   3. The current iteration is not a mult...
        #      we compute the unconditional branch...
        is_within_timestep_range = (
            self.unconditional_batch_timestep_skip_range[0]
            < self.current_timestep_callback()
            < self.unconditional_batch_timestep_skip_range[1]
        )
        should_skip_uncond = (
            self.state.iteration > 0
            and is_within_timestep_range
            and self.state.iteration % self.unconditional_batch_skip_range != 0
            and not self.is_guidance_distilled
        )

        if should_skip_uncond:
            is_any_kwarg_uncond = any(k in self.uncond_cond_input_kwargs_identifiers for k in kwargs.keys())
            if is_any_kwarg_uncond:
                logger.debug("FasterCache - Skipping unconditional branch computation")
                args = tuple([self._get_cond_input(arg) if torch.is_tensor(arg) else arg for arg in args])
                kwargs = {
                    k: v if k not in self.uncond_cond_input_kwargs_identifiers else self._get_cond_input(v)
                    for k, v in kwargs.items()
                }

        output = self.fn_ref.original_forward(*args, **kwargs)

        if self.is_guidance_distilled:
            self.state.iteration += 1
            return output

        if torch.is_tensor(output):
            hidden_states = output
        elif isinstance(output, (tuple, Transformer2DModelOutput)):
            hidden_states = output[0]

        batch_size = hidden_states.size(0)

        if should_skip_uncond:
            self.state.low_frequency_delta = self.state.low_frequency_delta * self.low_frequency_weight_callback(
                module
            )
            self.state.high_frequency_delta = self.state.high_frequency_delta * self.high_frequency_weight_callback(
                module
            )

            if self.tensor_format == "BCFHW":
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                hidden_states = hidden_states.flatten(0, 1)

            low_freq_cond, high_freq_cond = _split_low_high_freq(hidden_states.float())

            # Approximate/compute the unconditiona...
            low_freq_uncond = self.state.low_frequency_delta + low_freq_cond
            high_freq_uncond = self.state.high_frequency_delta + high_freq_cond
            uncond_freq = low_freq_uncond + high_freq_uncond

            uncond_states = torch.fft.ifftshift(uncond_freq)
            uncond_states = torch.fft.ifft2(uncond_states).real

            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                uncond_states = uncond_states.unflatten(0, (batch_size, -1))
                hidden_states = hidden_states.unflatten(0, (batch_size, -1))
            if self.tensor_format == "BCFHW":
                uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

            # Concatenate the approximated unconditional and predicted conditional branches
            uncond_states = uncond_states.to(hidden_states.dtype)
            hidden_states = torch.cat([uncond_states, hidden_states], dim=0)
        else:
            uncond_states, cond_states = hidden_states.chunk(2, dim=0)
            if self.tensor_format == "BCFHW":
                uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
                cond_states = cond_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                uncond_states = uncond_states.flatten(0, 1)
                cond_states = cond_states.flatten(0, 1)

            low_freq_uncond, high_freq_uncond = _split_low_high_freq(uncond_states.float())
            low_freq_cond, high_freq_cond = _split_low_high_freq(cond_states.float())
            self.state.low_frequency_delta = low_freq_uncond - low_freq_cond
            self.state.high_frequency_delta = high_freq_uncond - high_freq_cond

        self.state.iteration += 1
        if torch.is_tensor(output):
            output = hidden_states
        elif isinstance(output, tuple):
            output = (hidden_states, *output[1:])
        else:
            output.sample = hidden_states

        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state.reset()
        return module

class FasterCacheBlockHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        block_skip_range: int,
        timestep_skip_range: Tuple[int, int],
        is_guidance_distilled: bool,
        weight_callback: Callable[[torch.nn.Module], float],
        current_timestep_callback: Callable[[], int],
    ) -> None:
        super().__init__()

        self.block_skip_range = block_skip_range
        self.timestep_skip_range = timestep_skip_range
        self.is_guidance_distilled = is_guidance_distilled

        self.weight_callback = weight_callback
        self.current_timestep_callback = current_timestep_callback

    def initialize_hook(self, module):
        self.state = FasterCacheBlockState()
        return module

    def _compute_approximated_attention_output(
        self, t_2_output: torch.Tensor, t_output: torch.Tensor, weight: float, batch_size: int
    ) -> torch.Tensor:
        if t_2_output.size(0) != batch_size:
            # The cache t_2_output contains both b...
            # take the conditional branch outputs.
            assert t_2_output.size(0) == 2 * batch_size
            t_2_output = t_2_output[batch_size:]
        if t_output.size(0) != batch_size:
            # The cache t_output contains both bat...
            # take the conditional branch outputs.
            assert t_output.size(0) == 2 * batch_size
            t_output = t_output[batch_size:]
        return t_output + (t_output - t_2_output) * weight

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        batch_size = [
            *[arg.size(0) for arg in args if torch.is_tensor(arg)],
            *[v.size(0) for v in kwargs.values() if torch.is_tensor(v)],
        ][0]
        if self.state.batch_size is None:
            # Will be updated on first forward pass through the denoiser
            self.state.batch_size = batch_size

        # If we have to skip due to the skip conditions, then let's skip as expected.
        # But, we can't skip if the denoiser wants...
        # is because the expected output shapes of...
        # the cache (which only caches conditional...
        # unconditional-conditional batch size) is...
        # skip. Otherwise, we conditionally skip t...
        is_within_timestep_range = (
            self.timestep_skip_range[0] < self.current_timestep_callback() < self.timestep_skip_range[1]
        )
        if not is_within_timestep_range:
            should_skip_attention = False
        else:
            should_compute_attention = self.state.iteration > 0 and self.state.iteration % self.block_skip_range == 0
            should_skip_attention = not should_compute_attention
        if should_skip_attention:
            should_skip_attention = self.is_guidance_distilled or self.state.batch_size != batch_size

        if should_skip_attention:
            logger.debug("FasterCache - Skipping attention and using approximation")
            if torch.is_tensor(self.state.cache[-1]):
                t_2_output, t_output = self.state.cache
                weight = self.weight_callback(module)
                output = self._compute_approximated_attention_output(t_2_output, t_output, weight, batch_size)
            else:
                # The cache contains multiple tens...
                # Diffusers blocks can return mult...
                # In our cache, we would have [[A_...
                # a forward pass of the block. We...
                # The zip(*state.cache) operation...
                # allows us to compute the approxi...
                output = ()
                for t_2_output, t_output in zip(*self.state.cache):
                    result = self._compute_approximated_attention_output(
                        t_2_output, t_output, self.weight_callback(module), batch_size
                    )
                    output += (result,)
        else:
            logger.debug("FasterCache - Computing attention")
            output = self.fn_ref.original_forward(*args, **kwargs)

        # Note that the following condition for ge...
        # a single hidden_states tensor, or a tupl...
        # both cases.
        if torch.is_tensor(output):
            cache_output = output
            if not self.is_guidance_distilled and cache_output.size(0) == self.state.batch_size:
                # The output here can be both unco...
                # This is determined at the higher...
                cache_output = cache_output.chunk(2, dim=0)[1]
        else:
            # Cache all return values and perform the same operation as above
            cache_output = ()
            for out in output:
                if not self.is_guidance_distilled and out.size(0) == self.state.batch_size:
                    out = out.chunk(2, dim=0)[1]
                cache_output += (out,)

        if self.state.cache is None:
            self.state.cache = [cache_output, cache_output]
        else:
            self.state.cache = [self.state.cache[-1], cache_output]

        self.state.iteration += 1
        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state.reset()
        return module

def apply_faster_cache(module: torch.nn.Module, config: FasterCacheConfig) -> None:
    
    class FasterCacheDenoiserHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        unconditional_batch_skip_range: int,
        unconditional_batch_timestep_skip_range: Tuple[int, int],
        tensor_format: str,
        is_guidance_distilled: bool,
        uncond_cond_input_kwargs_identifiers: List[str],
        current_timestep_callback: Callable[[], int],
        low_frequency_weight_callback: Callable[[torch.nn.Module], torch.Tensor],
        high_frequency_weight_callback: Callable[[torch.nn.Module], torch.Tensor],
    ) -> None:
        super().__init__()

        self.unconditional_batch_skip_range = unconditional_batch_skip_range
        self.unconditional_batch_timestep_skip_range = unconditional_batch_timestep_skip_range
        # We can't easily detect what args are to...
        # can only do it for kwargs, hence they ar...
        # If a model is to be made compatible with...
        # contain batchwise-concatenated unconditional and conditional inputs are passed as kwargs.
        self.uncond_cond_input_kwargs_identifiers = uncond_cond_input_kwargs_identifiers
        self.tensor_format = tensor_format
        self.is_guidance_distilled = is_guidance_distilled

        self.current_timestep_callback = current_timestep_callback
        self.low_frequency_weight_callback = low_frequency_weight_callback
        self.high_frequency_weight_callback = high_frequency_weight_callback

    def initialize_hook(self, module):
        self.state = FasterCacheDenoiserState()
        return module

    @staticmethod
    def _get_cond_input(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this method assumes that the input...
        # followed by conditional inputs.
        _, cond = input.chunk(2, dim=0)
        return cond

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        # Split the unconditional and conditional...
        # requirements for skipping the unconditional branch are met as described in the paper.
        # We skip the unconditional branch only if the following conditions are met:
        #   1. We have completed at least one iteration of the denoiser
        #   2. The current timestep is within the...
        #      where approximating the uncondition...
        #      without a significant loss in quality.
        #   3. The current iteration is not a mult...
        #      we compute the unconditional branch...
        is_within_timestep_range = (
            self.unconditional_batch_timestep_skip_range[0]
            < self.current_timestep_callback()
            < self.unconditional_batch_timestep_skip_range[1]
        )
        should_skip_uncond = (
            self.state.iteration > 0
            and is_within_timestep_range
            and self.state.iteration % self.unconditional_batch_skip_range != 0
            and not self.is_guidance_distilled
        )

        if should_skip_uncond:
            is_any_kwarg_uncond = any(k in self.uncond_cond_input_kwargs_identifiers for k in kwargs.keys())
            if is_any_kwarg_uncond:
                logger.debug("FasterCache - Skipping unconditional branch computation")
                args = tuple([self._get_cond_input(arg) if torch.is_tensor(arg) else arg for arg in args])
                kwargs = {
                    k: v if k not in self.uncond_cond_input_kwargs_identifiers else self._get_cond_input(v)
                    for k, v in kwargs.items()
                }

        output = self.fn_ref.original_forward(*args, **kwargs)

        if self.is_guidance_distilled:
            self.state.iteration += 1
            return output

        if torch.is_tensor(output):
            hidden_states = output
        elif isinstance(output, (tuple, Transformer2DModelOutput)):
            hidden_states = output[0]

        batch_size = hidden_states.size(0)

        if should_skip_uncond:
            self.state.low_frequency_delta = self.state.low_frequency_delta * self.low_frequency_weight_callback(
                module
            )
            self.state.high_frequency_delta = self.state.high_frequency_delta * self.high_frequency_weight_callback(
                module
            )

            if self.tensor_format == "BCFHW":
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                hidden_states = hidden_states.flatten(0, 1)

            low_freq_cond, high_freq_cond = _split_low_high_freq(hidden_states.float())

            # Approximate/compute the unconditiona...
            low_freq_uncond = self.state.low_frequency_delta + low_freq_cond
            high_freq_uncond = self.state.high_frequency_delta + high_freq_cond
            uncond_freq = low_freq_uncond + high_freq_uncond

            uncond_states = torch.fft.ifftshift(uncond_freq)
            uncond_states = torch.fft.ifft2(uncond_states).real

            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                uncond_states = uncond_states.unflatten(0, (batch_size, -1))
                hidden_states = hidden_states.unflatten(0, (batch_size, -1))
            if self.tensor_format == "BCFHW":
                uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

            # Concatenate the approximated unconditional and predicted conditional branches
            uncond_states = uncond_states.to(hidden_states.dtype)
            hidden_states = torch.cat([uncond_states, hidden_states], dim=0)
        else:
            uncond_states, cond_states = hidden_states.chunk(2, dim=0)
            if self.tensor_format == "BCFHW":
                uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
                cond_states = cond_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                uncond_states = uncond_states.flatten(0, 1)
                cond_states = cond_states.flatten(0, 1)

            low_freq_uncond, high_freq_uncond = _split_low_high_freq(uncond_states.float())
            low_freq_cond, high_freq_cond = _split_low_high_freq(cond_states.float())
            self.state.low_frequency_delta = low_freq_uncond - low_freq_cond
            self.state.high_frequency_delta = high_freq_uncond - high_freq_cond

        self.state.iteration += 1
        if torch.is_tensor(output):
            output = hidden_states
        elif isinstance(output, tuple):
            output = (hidden_states, *output[1:])
        else:
            output.sample = hidden_states

        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state.reset()
        return module

class FasterCacheBlockHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        block_skip_range: int,
        timestep_skip_range: Tuple[int, int],
        is_guidance_distilled: bool,
        weight_callback: Callable[[torch.nn.Module], float],
        current_timestep_callback: Callable[[], int],
    ) -> None:
        super().__init__()

        self.block_skip_range = block_skip_range
        self.timestep_skip_range = timestep_skip_range
        self.is_guidance_distilled = is_guidance_distilled

        self.weight_callback = weight_callback
        self.current_timestep_callback = current_timestep_callback

    def initialize_hook(self, module):
        self.state = FasterCacheBlockState()
        return module

    def _compute_approximated_attention_output(
        self, t_2_output: torch.Tensor, t_output: torch.Tensor, weight: float, batch_size: int
    ) -> torch.Tensor:
        if t_2_output.size(0) != batch_size:
            # The cache t_2_output contains both b...
            # take the conditional branch outputs.
            assert t_2_output.size(0) == 2 * batch_size
            t_2_output = t_2_output[batch_size:]
        if t_output.size(0) != batch_size:
            # The cache t_output contains both bat...
            # take the conditional branch outputs.
            assert t_output.size(0) == 2 * batch_size
            t_output = t_output[batch_size:]
        return t_output + (t_output - t_2_output) * weight

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        batch_size = [
            *[arg.size(0) for arg in args if torch.is_tensor(arg)],
            *[v.size(0) for v in kwargs.values() if torch.is_tensor(v)],
        ][0]
        if self.state.batch_size is None:
            # Will be updated on first forward pass through the denoiser
            self.state.batch_size = batch_size

        # If we have to skip due to the skip conditions, then let's skip as expected.
        # But, we can't skip if the denoiser wants...
        # is because the expected output shapes of...
        # the cache (which only caches conditional...
        # unconditional-conditional batch size) is...
        # skip. Otherwise, we conditionally skip t...
        is_within_timestep_range = (
            self.timestep_skip_range[0] < self.current_timestep_callback() < self.timestep_skip_range[1]
        )
        if not is_within_timestep_range:
            should_skip_attention = False
        else:
            should_compute_attention = self.state.iteration > 0 and self.state.iteration % self.block_skip_range == 0
            should_skip_attention = not should_compute_attention
        if should_skip_attention:
            should_skip_attention = self.is_guidance_distilled or self.state.batch_size != batch_size

        if should_skip_attention:
            logger.debug("FasterCache - Skipping attention and using approximation")
            if torch.is_tensor(self.state.cache[-1]):
                t_2_output, t_output = self.state.cache
                weight = self.weight_callback(module)
                output = self._compute_approximated_attention_output(t_2_output, t_output, weight, batch_size)
            else:
                # The cache contains multiple tens...
                # Diffusers blocks can return mult...
                # In our cache, we would have [[A_...
                # a forward pass of the block. We...
                # The zip(*state.cache) operation...
                # allows us to compute the approxi...
                output = ()
                for t_2_output, t_output in zip(*self.state.cache):
                    result = self._compute_approximated_attention_output(
                        t_2_output, t_output, self.weight_callback(module), batch_size
                    )
                    output += (result,)
        else:
            logger.debug("FasterCache - Computing attention")
            output = self.fn_ref.original_forward(*args, **kwargs)

        # Note that the following condition for ge...
        # a single hidden_states tensor, or a tupl...
        # both cases.
        if torch.is_tensor(output):
            cache_output = output
            if not self.is_guidance_distilled and cache_output.size(0) == self.state.batch_size:
                # The output here can be both unco...
                # This is determined at the higher...
                cache_output = cache_output.chunk(2, dim=0)[1]
        else:
            # Cache all return values and perform the same operation as above
            cache_output = ()
            for out in output:
                if not self.is_guidance_distilled and out.size(0) == self.state.batch_size:
                    out = out.chunk(2, dim=0)[1]
                cache_output += (out,)

        if self.state.cache is None:
            self.state.cache = [cache_output, cache_output]
        else:
            self.state.cache = [self.state.cache[-1], cache_output]

        self.state.iteration += 1
        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state.reset()
        return module

def apply_faster_cache(module: torch.nn.Module, config: FasterCacheConfig) -> None:
    class FasterCacheDenoiserHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        unconditional_batch_skip_range: int,
        unconditional_batch_timestep_skip_range: Tuple[int, int],
        tensor_format: str,
        is_guidance_distilled: bool,
        uncond_cond_input_kwargs_identifiers: List[str],
        current_timestep_callback: Callable[[], int],
        low_frequency_weight_callback: Callable[[torch.nn.Module], torch.Tensor],
        high_frequency_weight_callback: Callable[[torch.nn.Module], torch.Tensor],
    ) -> None:
        super().__init__()

        self.unconditional_batch_skip_range = unconditional_batch_skip_range
        self.unconditional_batch_timestep_skip_range = unconditional_batch_timestep_skip_range
        # We can't easily detect what args are to...
        # can only do it for kwargs, hence they ar...
        # If a model is to be made compatible with...
        # contain batchwise-concatenated unconditional and conditional inputs are passed as kwargs.
        self.uncond_cond_input_kwargs_identifiers = uncond_cond_input_kwargs_identifiers
        self.tensor_format = tensor_format
        self.is_guidance_distilled = is_guidance_distilled

        self.current_timestep_callback = current_timestep_callback
        self.low_frequency_weight_callback = low_frequency_weight_callback
        self.high_frequency_weight_callback = high_frequency_weight_callback

    def initialize_hook(self, module):
        self.state = FasterCacheDenoiserState()
        return module

    @staticmethod
    def _get_cond_input(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this method assumes that the input...
        # followed by conditional inputs.
        _, cond = input.chunk(2, dim=0)
        return cond

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        # Split the unconditional and conditional...
        # requirements for skipping the unconditional branch are met as described in the paper.
        # We skip the unconditional branch only if the following conditions are met:
        #   1. We have completed at least one iteration of the denoiser
        #   2. The current timestep is within the...
        #      where approximating the uncondition...
        #      without a significant loss in quality.
        #   3. The current iteration is not a mult...
        #      we compute the unconditional branch...
        is_within_timestep_range = (
            self.unconditional_batch_timestep_skip_range[0]
            < self.current_timestep_callback()
            < self.unconditional_batch_timestep_skip_range[1]
        )
        should_skip_uncond = (
            self.state.iteration > 0
            and is_within_timestep_range
            and self.state.iteration % self.unconditional_batch_skip_range != 0
            and not self.is_guidance_distilled
        )

        if should_skip_uncond:
            is_any_kwarg_uncond = any(k in self.uncond_cond_input_kwargs_identifiers for k in kwargs.keys())
            if is_any_kwarg_uncond:
                logger.debug("FasterCache - Skipping unconditional branch computation")
                args = tuple([self._get_cond_input(arg) if torch.is_tensor(arg) else arg for arg in args])
                kwargs = {
                    k: v if k not in self.uncond_cond_input_kwargs_identifiers else self._get_cond_input(v)
                    for k, v in kwargs.items()
                }

        output = self.fn_ref.original_forward(*args, **kwargs)

        if self.is_guidance_distilled:
            self.state.iteration += 1
            return output

        if torch.is_tensor(output):
            hidden_states = output
        elif isinstance(output, (tuple, Transformer2DModelOutput)):
            hidden_states = output[0]

        batch_size = hidden_states.size(0)

        if should_skip_uncond:
            self.state.low_frequency_delta = self.state.low_frequency_delta * self.low_frequency_weight_callback(
                module
            )
            self.state.high_frequency_delta = self.state.high_frequency_delta * self.high_frequency_weight_callback(
                module
            )

            if self.tensor_format == "BCFHW":
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                hidden_states = hidden_states.flatten(0, 1)

            low_freq_cond, high_freq_cond = _split_low_high_freq(hidden_states.float())

            # Approximate/compute the unconditiona...
            low_freq_uncond = self.state.low_frequency_delta + low_freq_cond
            high_freq_uncond = self.state.high_frequency_delta + high_freq_cond
            uncond_freq = low_freq_uncond + high_freq_uncond

            uncond_states = torch.fft.ifftshift(uncond_freq)
            uncond_states = torch.fft.ifft2(uncond_states).real

            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                uncond_states = uncond_states.unflatten(0, (batch_size, -1))
                hidden_states = hidden_states.unflatten(0, (batch_size, -1))
            if self.tensor_format == "BCFHW":
                uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

            # Concatenate the approximated unconditional and predicted conditional branches
            uncond_states = uncond_states.to(hidden_states.dtype)
            hidden_states = torch.cat([uncond_states, hidden_states], dim=0)
        else:
            uncond_states, cond_states = hidden_states.chunk(2, dim=0)
            if self.tensor_format == "BCFHW":
                uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
                cond_states = cond_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                uncond_states = uncond_states.flatten(0, 1)
                cond_states = cond_states.flatten(0, 1)

            low_freq_uncond, high_freq_uncond = _split_low_high_freq(uncond_states.float())
            low_freq_cond, high_freq_cond = _split_low_high_freq(cond_states.float())
            self.state.low_frequency_delta = low_freq_uncond - low_freq_cond
            self.state.high_frequency_delta = high_freq_uncond - high_freq_cond

        self.state.iteration += 1
        if torch.is_tensor(output):
            output = hidden_states
        elif isinstance(output, tuple):
            output = (hidden_states, *output[1:])
        else:
            output.sample = hidden_states

        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state.reset()
        return module

class FasterCacheBlockHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        block_skip_range: int,
        timestep_skip_range: Tuple[int, int],
        is_guidance_distilled: bool,
        weight_callback: Callable[[torch.nn.Module], float],
        current_timestep_callback: Callable[[], int],
    ) -> None:
        super().__init__()

        self.block_skip_range = block_skip_range
        self.timestep_skip_range = timestep_skip_range
        self.is_guidance_distilled = is_guidance_distilled

        self.weight_callback = weight_callback
        self.current_timestep_callback = current_timestep_callback

    def initialize_hook(self, module):
        self.state = FasterCacheBlockState()
        return module

    def _compute_approximated_attention_output(
        self, t_2_output: torch.Tensor, t_output: torch.Tensor, weight: float, batch_size: int
    ) -> torch.Tensor:
        if t_2_output.size(0) != batch_size:
            # The cache t_2_output contains both b...
            # take the conditional branch outputs.
            assert t_2_output.size(0) == 2 * batch_size
            t_2_output = t_2_output[batch_size:]
        if t_output.size(0) != batch_size:
            # The cache t_output contains both bat...
            # take the conditional branch outputs.
            assert t_output.size(0) == 2 * batch_size
            t_output = t_output[batch_size:]
        return t_output + (t_output - t_2_output) * weight

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        batch_size = [
            *[arg.size(0) for arg in args if torch.is_tensor(arg)],
            *[v.size(0) for v in kwargs.values() if torch.is_tensor(v)],
        ][0]
        if self.state.batch_size is None:
            # Will be updated on first forward pass through the denoiser
            self.state.batch_size = batch_size

        # If we have to skip due to the skip conditions, then let's skip as expected.
        # But, we can't skip if the denoiser wants...
        # is because the expected output shapes of...
        # the cache (which only caches conditional...
        # unconditional-conditional batch size) is...
        # skip. Otherwise, we conditionally skip t...
        is_within_timestep_range = (
            self.timestep_skip_range[0] < self.current_timestep_callback() < self.timestep_skip_range[1]
        )
        if not is_within_timestep_range:
            should_skip_attention = False
        else:
            should_compute_attention = self.state.iteration > 0 and self.state.iteration % self.block_skip_range == 0
            should_skip_attention = not should_compute_attention
        if should_skip_attention:
            should_skip_attention = self.is_guidance_distilled or self.state.batch_size != batch_size

        if should_skip_attention:
            logger.debug("FasterCache - Skipping attention and using approximation")
            if torch.is_tensor(self.state.cache[-1]):
                t_2_output, t_output = self.state.cache
                weight = self.weight_callback(module)
                output = self._compute_approximated_attention_output(t_2_output, t_output, weight, batch_size)
            else:
                # The cache contains multiple tens...
                # Diffusers blocks can return mult...
                # In our cache, we would have [[A_...
                # a forward pass of the block. We...
                # The zip(*state.cache) operation...
                # allows us to compute the approxi...
                output = ()
                for t_2_output, t_output in zip(*self.state.cache):
                    result = self._compute_approximated_attention_output(
                        t_2_output, t_output, self.weight_callback(module), batch_size
                    )
                    output += (result,)
        else:
            logger.debug("FasterCache - Computing attention")
            output = self.fn_ref.original_forward(*args, **kwargs)

        # Note that the following condition for ge...
        # a single hidden_states tensor, or a tupl...
        # both cases.
        if torch.is_tensor(output):
            cache_output = output
            if not self.is_guidance_distilled and cache_output.size(0) == self.state.batch_size:
                # The output here can be both unco...
                # This is determined at the higher...
                cache_output = cache_output.chunk(2, dim=0)[1]
        else:
            # Cache all return values and perform the same operation as above
            cache_output = ()
            for out in output:
                if not self.is_guidance_distilled and out.size(0) == self.state.batch_size:
                    out = out.chunk(2, dim=0)[1]
                cache_output += (out,)

        if self.state.cache is None:
            self.state.cache = [cache_output, cache_output]
        else:
            self.state.cache = [self.state.cache[-1], cache_output]

        self.state.iteration += 1
        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state.reset()
        return module

def apply_faster_cache(module: torch.nn.Module, config: FasterCacheConfig) -> None:


    logger.warning(
        "FasterCache is a purely experimental feature and may not work as expected. Not all models support FasterCache. "
        "The API is subject to change in future releases, with no guarantee of backward compatibility. Please report any issues at "
        "https://github.com/huggingface/diffusers/issues."
    )

    if config.attention_weight_callback is None:
        # If the user has not provided a weight callback, we default to 0.5 for all timesteps.
        # In the paper, they recommend using a gra...
        # this depends from model-to-model. It is...
        # use a different weight function. Defaulting to 0.5 works well in practice for most cases.
        logger.warning(
            "No `attention_weight_callback` provided when enabling FasterCache. Defaulting to using a weight of 0.5 for all timesteps."
        )
        config.attention_weight_callback = lambda _: 0.5

    if config.low_frequency_weight_callback is None:
        logger.debug(
            "Low frequency weight callback not provided when enabling FasterCache. Defaulting to behaviour described in the paper."
        )

        def low_frequency_weight_callback(module: torch.nn.Module) -> float:
            is_within_range = (
                config.low_frequency_weight_update_timestep_range[0]
                < config.current_timestep_callback()
                < config.low_frequency_weight_update_timestep_range[1]
            )
            return config.alpha_low_frequency if is_within_range else 1.0

        config.low_frequency_weight_callback = low_frequency_weight_callback

    if config.high_frequency_weight_callback is None:
        logger.debug(
            "High frequency weight callback not provided when enabling FasterCache. Defaulting to behaviour described in the paper."
        )

        def high_frequency_weight_callback(module: torch.nn.Module) -> float:
            is_within_range = (
                config.high_frequency_weight_update_timestep_range[0]
                < config.current_timestep_callback()
                < config.high_frequency_weight_update_timestep_range[1]
            )
            return config.alpha_high_frequency if is_within_range else 1.0

        config.high_frequency_weight_callback = high_frequency_weight_callback

    supported_tensor_formats = ["BCFHW", "BFCHW", "BCHW"]  # TODO(aryan): Support BSC for LTX Video
    if config.tensor_format not in supported_tensor_formats:
        raise ValueError(f"`tensor_format` must be one of {supported_tensor_formats}, but got {config.tensor_format}.")

    _apply_faster_cache_on_denoiser(module, config)

    for name, submodule in module.named_modules():
        if not isinstance(submodule, _ATTENTION_CLASSES):
            continue
        if any(re.search(identifier, name) is not None for identifier in _TRANSFORMER_BLOCK_IDENTIFIERS):
            _apply_faster_cache_on_attention_class(name, submodule, config)

def _apply_faster_cache_on_denoiser(module: torch.nn.Module, config: FasterCacheConfig) -> None:
    hook = FasterCacheDenoiserHook(
        config.unconditional_batch_skip_range,
        config.unconditional_batch_timestep_skip_range,
        config.tensor_format,
        config.is_guidance_distilled,
        config._unconditional_conditional_input_kwargs_identifiers,
        config.current_timestep_callback,
        config.low_frequency_weight_callback,
        config.high_frequency_weight_callback,
    )
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.register_hook(hook, _FASTER_CACHE_DENOISER_HOOK)

def _apply_faster_cache_on_attention_class(name: str, module: AttentionModuleMixin, config: FasterCacheConfig) -> None:
    is_spatial_self_attention = (
        any(re.search(identifier, name) is not None for identifier in config.spatial_attention_block_identifiers)
        and config.spatial_attention_block_skip_range is not None
        and not getattr(module, "is_cross_attention", False)
    )
    is_temporal_self_attention = (
        any(re.search(identifier, name) is not None for identifier in config.temporal_attention_block_identifiers)
        and config.temporal_attention_block_skip_range is not None
        and not module.is_cross_attention
    )

    block_skip_range, timestep_skip_range, block_type = None, None, None
    if is_spatial_self_attention:
        block_skip_range = config.spatial_attention_block_skip_range
        timestep_skip_range = config.spatial_attention_timestep_skip_range
        block_type = "spatial"
    elif is_temporal_self_attention:
        block_skip_range = config.temporal_attention_block_skip_range
        timestep_skip_range = config.temporal_attention_timestep_skip_range
        block_type = "temporal"

    if block_skip_range is None or timestep_skip_range is None:
        logger.debug(
            f'Unable to apply FasterCache to the selected layer: "{name}" because it does '
            f"not match any of the required criteria for spatial or temporal attention layers. Note, "
            f"however, that this layer may still be valid for applying PAB. Please specify the correct "
            f"block identifiers in the configuration or use the specialized `apply_faster_cache_on_module` "
            f"function to apply FasterCache to this layer."
        )
        return

    logger.debug(f"Enabling FasterCache ({block_type}) for layer: {name}")
    hook = FasterCacheBlockHook(
        block_skip_range,
        timestep_skip_range,
        config.is_guidance_distilled,
        config.attention_weight_callback,
        config.current_timestep_callback,
    )
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.register_hook(hook, _FASTER_CACHE_BLOCK_HOOK)

# Reference: https://github.com/Vchitect/FasterCac...
@torch.no_grad()
def _split_low_high_freq(x):
    fft = torch.fft.fft2(x)
    fft_shifted = torch.fft.fftshift(fft)
    height, width = x.shape[-2:]
    radius = min(height, width) // 5

    y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width))
    center_x, center_y = width // 2, height // 2
    mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius**2

    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(x.device)
    high_freq_mask = ~low_freq_mask

    low_freq_fft = fft_shifted * low_freq_mask
    high_freq_fft = fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft
