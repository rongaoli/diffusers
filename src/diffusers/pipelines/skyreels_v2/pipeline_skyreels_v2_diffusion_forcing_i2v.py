import html
import math
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ftfy
import PIL
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import SkyReelsV2LoraLoaderMixin
from ...models import AutoencoderKLWan, SkyReelsV2Transformer3DModel
from ...schedulers import UniPCMultistepScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import SkyReelsV2PipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """\


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

# Copied from diffusers.pipelines.stable_diffusion...
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class SkyReelsV2DiffusionForcingImageToVideoPipeline(DiffusionPipeline, SkyReelsV2LoraLoaderMixin):


    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: SkyReelsV2Transformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        
        """r"""
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        overlap_history=None,
        num_frames=None,
        base_num_frames=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `image_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if num_frames > base_num_frames and overlap_history is None:
            raise ValueError(
                "`overlap_history` is required when `num_frames` exceeds `base_num_frames` to ensure smooth transitions in long video generation. "
                "Please specify a value for `overlap_history`. Recommended values are 17 or 37."
            )

    def prepare_latents(
        self,
        image: Optional[PipelineImageInput],
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 97,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        video_latents: Optional[torch.Tensor] = None,
        base_latent_num_frames: Optional[int] = None,
        causal_block_size: Optional[int] = None,
        overlap_history_latent_frames: Optional[int] = None,
        long_video_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        prefix_video_latents_frames = 0

        if video_latents is not None:  # long video generation at the iterations other than the first one
            condition = video_latents[:, :, -overlap_history_latent_frames:]

            if condition.shape[2] % causal_block_size != 0:
                truncate_len_latents = condition.shape[2] % causal_block_size
                logger.warning(
                    f"The length of prefix video latents is truncated by {truncate_len_latents} frames for the causal block size alignment. "
                    f"This truncation ensures compatibility with the causal block size, which is required for proper processing. "
                    f"However, it may slightly affect the continuity of the generated video at the truncation boundary."
                )
                condition = condition[:, :, :-truncate_len_latents]
            prefix_video_latents_frames = condition.shape[2]

            finished_frame_num = (
                long_video_iter * (base_latent_num_frames - overlap_history_latent_frames)
                + overlap_history_latent_frames
            )
            left_frame_num = num_latent_frames - finished_frame_num
            num_latent_frames = min(left_frame_num + overlap_history_latent_frames, base_latent_num_frames)
        elif base_latent_num_frames is not None:  # long video generation at the first iteration
            num_latent_frames = base_latent_num_frames
        else:  # short video generation
            num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        if image is not None:
            image = image.unsqueeze(2)
            if last_image is not None:
                last_image = last_image.unsqueeze(2)
                video_condition = torch.cat([image, last_image], dim=0)
            else:
                video_condition = image

            video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )

            if isinstance(generator, list):
                latent_condition = [
                    retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
                ]
                latent_condition = torch.cat(latent_condition)
            else:
                latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
                latent_condition = latent_condition.repeat_interleave(batch_size, dim=0)

            latent_condition = latent_condition.to(dtype)
            condition = (latent_condition - latents_mean) * latents_std
            prefix_video_latents_frames = condition.shape[2]

        return latents, num_latent_frames, condition, prefix_video_latents_frames

    # Copied from diffusers.pipelines.skyreels_v2....
    def generate_timestep_matrix(
        self,
        num_latent_frames: int,
        step_template: torch.Tensor,
        base_num_latent_frames: int,
        ar_step: int = 5,
        num_pre_ready: int = 0,
        causal_block_size: int = 1,
        shrink_interval_with_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        This function implements the core diffusion forcing algorithm that creates a coordinated denoising schedule
        across temporal frames. It supports both synchronous and asynchronous generation modes:

        **Synchronous Mode** (ar_step=0, causal_block_size=1):
        - All frames are denoised simultaneously at each timestep
        - Each frame follows the same denoising trajectory: [1000, 800, 600, ..., 0]
        - Simpler but may have less temporal consistency for long videos

        **Asynchronous Mode** (ar_step>0, causal_block_size>1):
        - Frames are grouped into causal blocks and processed block/chunk-wise
        - Each block is denoised in a staggered pattern creating a "denoising wave"
        - Earlier blocks are more denoised, later blocks lag behind by ar_step timesteps
        - Creates stronger temporal dependencies and better consistency
                - step_matrix (torch.Tensor): Matrix of timesteps for each frame at each iteration Shape:
                  [num_iterations, num_latent_frames]
                - step_index (torch.Tensor): Index matrix for timestep lookup Shape: [num_iterations,
                  num_latent_frames]
                - step_update_mask (torch.Tensor): Boolean mask indicating which frames to update Shape:
                  [num_iterations, num_latent_frames]
                - valid_interval (list[tuple]): List of (start, end) intervals for each iteration
