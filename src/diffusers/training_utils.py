import contextlib
import copy
import gc
import math
import random
import re
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

import numpy as np
import torch

if getattr(torch, "distributed", None) is not None:
    from torch.distributed.fsdp import CPUOffload, ShardingStrategy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .models import UNet2DConditionModel
from .pipelines import DiffusionPipeline
from .schedulers import SchedulerMixin
from .utils import (
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    deprecate,
    is_accelerate_available,
    is_peft_available,
    is_torch_npu_available,
    is_torchvision_available,
    is_transformers_available,
)

if is_transformers_available():
    import transformers

    if transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
        import deepspeed

if is_accelerate_available():
    from accelerate.logging import get_logger

if is_peft_available():
    from peft import set_peft_model_state_dict

if is_torchvision_available():
    from torchvision import transforms

if is_torch_npu_available():
    import torch_npu  # noqa: F401

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_torch_npu_available():
        torch.npu.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

def compute_snr(noise_scheduler, timesteps):

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/...
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def resolve_interpolation_mode(interpolation_type: str):

    Implements "DREAM (Diffusion Rectification and Estimation-Adaptive Models)" from
    https://huggingface.co/papers/2312.00210. DREAM helps align training with sampling to help training be more
    efficient and accurate at the cost of an extra forward step without gradients.
        `unet`: The state unet to use to make a prediction.
        `noise_scheduler`: The noise scheduler used to add noise for the given timestep.
        `timesteps`: The timesteps for the noise_scheduler to user.
        `noise`: A tensor of noise in the shape of noisy_latents.
        `noisy_latents`: Previously noise latents from the training loop.
        `target`: The ground-truth tensor to predict after eps is removed.
        `encoder_hidden_states`: Text embeddings from the text model.
        `dream_detail_preservation`: A float value that indicates detail preservation level.
          See reference.
):


    text_encoder_state_dict = {
        f"{k.replace(prefix, '')}": v for k, v in lora_state_dict.items() if k.startswith(prefix)
    }
    text_encoder_state_dict = convert_state_dict_to_peft(convert_state_dict_to_diffusers(text_encoder_state_dict))
    set_peft_model_state_dict(text_encoder, text_encoder_state_dict, adapter_name="default")

def _collate_lora_metadata(modules_to_save: Dict[str, torch.nn.Module]) -> Dict[str, Any]:
    metadatas = {}
    for module_name, module in modules_to_save.items():
        if module is not None:
            metadatas[f"{module_name}_lora_adapter_metadata"] = module.peft_config["default"].to_dict()
    return metadatas

def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    device: Union[torch.device, str] = "cpu",
    generator: Optional[torch.Generator] = None,
):

    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):

    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

def free_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif is_torch_npu_available():
        torch_npu.npu.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

@contextmanager
def offload_models(
    *modules: Union[torch.nn.Module, DiffusionPipeline], device: Union[str, torch.device], offload: bool = True
):

    if offload:
        is_model = not any(isinstance(m, DiffusionPipeline) for m in modules)
        # record where each module was
        if is_model:
            original_devices = [next(m.parameters()).device for m in modules]
        else:
            assert len(modules) == 1
            # For DiffusionPipeline, wrap the device in a list to make it iterable
            original_devices = [modules[0].device]
        # move to target device
        for m in modules:
            m.to(device)

    try:
        yield
    finally:
        if offload:
            # move back to original devices
            for m, orig_dev in zip(modules, original_devices):
                m.to(orig_dev)

def parse_buckets_string(buckets_str):

    if not buckets_str:
        raise ValueError("Bucket string cannot be empty.")

    bucket_pairs = buckets_str.strip().split(";")
    parsed_buckets = []
    for pair_str in bucket_pairs:
        match = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*$", pair_str)
        if not match:
            raise ValueError(f"Invalid bucket format: '{pair_str}'. Expected 'height,width'.")
        try:
            height = int(match.group(1))
            width = int(match.group(2))
            if height <= 0 or width <= 0:
                raise ValueError("Bucket dimensions must be positive integers.")
            if height % 8 != 0 or width % 8 != 0:
                warnings.warn(f"Bucket dimension ({height},{width}) not divisible by 8. This might cause issues.")
            parsed_buckets.append((height, width))
        except ValueError as e:
            raise ValueError(f"Invalid integer in bucket pair '{pair_str}': {e}") from e

    if not parsed_buckets:
        raise ValueError("No valid buckets found in the provided string.")

    return parsed_buckets

def find_nearest_bucket(h, w, bucket_options):

    min_metric = float("inf")
    best_bucket_idx = None
    for bucket_idx, (bucket_h, bucket_w) in enumerate(bucket_options):
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket_idx = bucket_idx
    return best_bucket_idx

def _to_cpu_contiguous(state_dicts) -> dict:
    return {k: v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v for k, v in state_dicts.items()}

def get_fsdp_kwargs_from_accelerator(accelerator) -> dict:


    kwargs = {}
    fsdp_state = getattr(accelerator.state, "fsdp_plugin", None)

    if fsdp_state is None:
        raise ValueError("Accelerate isn't configured to handle FSDP. Please update your installation.")

    fsdp_plugin = accelerator.state.fsdp_plugin

    if fsdp_plugin is None:
        # FSDP not enabled in Accelerator
        kwargs["sharding_strategy"] = ShardingStrategy.FULL_SHARD
    else:
        # FSDP is enabled → use plugin's strategy, or default if None
        kwargs["sharding_strategy"] = fsdp_plugin.sharding_strategy or ShardingStrategy.FULL_SHARD

    return kwargs

def wrap_with_fsdp(
    model: torch.nn.Module,
    device: Union[str, torch.device],
    offload: bool = True,
    use_orig_params: bool = True,
    limit_all_gathers: bool = True,
    fsdp_kwargs: Optional[Dict[str, Any]] = None,
    transformer_layer_cls: Optional[Set[Type[torch.nn.Module]]] = None,
) -> FSDP:


    logger = get_logger(__name__)

    if transformer_layer_cls is None:
        # Set the default layers if transformer_layer_cls is not provided
        transformer_layer_cls = type(model.model.language_model.layers[0])
        logger.info(f"transformer_layer_cls is not provided, auto-inferred as {transformer_layer_cls.__name__}")

    # Add auto-wrap policy if transformer layers specified
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls},
    )

    config = {
        "device_id": device,
        "cpu_offload": CPUOffload(offload_params=offload) if offload else None,
        "use_orig_params": use_orig_params,
        "limit_all_gathers": limit_all_gathers,
        "auto_wrap_policy": auto_wrap_policy,
    }

    if fsdp_kwargs:
        config.update(fsdp_kwargs)

    fsdp_model = FSDP(model, **config)
    return fsdp_model

# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    
    class EMAModel:


    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        foreach: bool = False,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):


        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

            # set use_ema_warmup to True if a torch.nn.Module is passed for backwards compatibility
            use_ema_warmup = True

        if kwargs.get("max_value", None) is not None:
            deprecation_message = "The `max_value` argument is deprecated. Please use `decay` instead."
            deprecate("max_value", "1.0.0", deprecation_message, standard_warn=False)
            decay = kwargs["max_value"]

        if kwargs.get("min_value", None) is not None:
            deprecation_message = "The `min_value` argument is deprecated. Please use `min_decay` instead."
            deprecate("min_value", "1.0.0", deprecation_message, standard_warn=False)
            min_decay = kwargs["min_value"]

        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        if kwargs.get("device", None) is not None:
            deprecation_message = "The `device` argument is deprecated. Please use `to` instead."
            deprecate("device", "1.0.0", deprecation_message, standard_warn=False)
            self.to(device=kwargs["device"])

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`
        self.foreach = foreach

        self.model_cls = model_cls
        self.model_config = model_config

    @classmethod
    def from_pretrained(cls, path, model_cls, foreach=False) -> "EMAModel":
        _, ema_kwargs = model_cls.from_config(path, return_unused_kwargs=True)
        model = model_cls.from_pretrained(path)

        ema_model = cls(model.parameters(), model_cls=model_cls, model_config=model.config, foreach=foreach)

        ema_model.load_state_dict(ema_kwargs)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        model = self.model_cls.from_config(self.model_config)
        state_dict = self.state_dict()
        state_dict.pop("shadow_params", None)

        model.register_to_config(**state_dict)
        self.copy_to(model.parameters())
        model.save_pretrained(path)

    def get_decay(self, optimization_step: int) -> float:

        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage.step` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage.step`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        context_manager = contextlib.nullcontext()

        if self.foreach:
            if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
                context_manager = deepspeed.zero.GatheredParameters(parameters, modifier_rank=None)

            with context_manager:
                params_grad = [param for param in parameters if param.requires_grad]
                s_params_grad = [
                    s_param for s_param, param in zip(self.shadow_params, parameters) if param.requires_grad
                ]

                if len(params_grad) < len(parameters):
                    torch._foreach_copy_(
                        [s_param for s_param, param in zip(self.shadow_params, parameters) if not param.requires_grad],
                        [param for param in parameters if not param.requires_grad],
                        non_blocking=True,
                    )

                torch._foreach_sub_(
                    s_params_grad, torch._foreach_sub(s_params_grad, params_grad), alpha=one_minus_decay
                )

        else:
            for s_param, param in zip(self.shadow_params, parameters):
                if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
                    context_manager = deepspeed.zero.GatheredParameters(param, modifier_rank=None)

                with context_manager:
                    if param.requires_grad:
                        s_param.sub_(one_minus_decay * (s_param - param))
                    else:
                        s_param.copy_(param)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:

        parameters = list(parameters)
        if self.foreach:
            torch._foreach_copy_(
                [param.data for param in parameters],
                [s_param.to(param.device).data for s_param, param in zip(self.shadow_params, parameters)],
            )
        else:
            for s_param, param in zip(self.shadow_params, parameters):
                param.data.copy_(s_param.to(param.device).data)

    def pin_memory(self) -> None:


        self.shadow_params = [p.pin_memory() for p in self.shadow_params]

    def to(self, device=None, dtype=None, non_blocking=False) -> None:

        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if p.is_floating_point()
            else p.to(device=device, non_blocking=non_blocking)
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:

        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": self.shadow_params,
        }

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:

        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:


        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights to `restore()`")
        if self.foreach:
            torch._foreach_copy_(
                [param.data for param in parameters], [c_param.data for c_param in self.temp_stored_params]
            )
        else:
            for c_param, param in zip(self.temp_stored_params, parameters):
                param.data.copy_(c_param.data)

        # Better memory-wise.
        self.temp_stored_params = None

    def load_state_dict(self, state_dict: dict) -> None:

        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        shadow_params = state_dict.get("shadow_params", None)
        if shadow_params is not None:
            self.shadow_params = shadow_params
            if not isinstance(self.shadow_params, list):
                raise ValueError("shadow_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
                raise ValueError("shadow_params must all be Tensors")
