import html
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import regex as re
import torch
from torch.nn import functional as F
from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...loaders import KandinskyLoraLoaderMixin
from ...models import AutoencoderKL
from ...models.transformers import Kandinsky5Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler

# Add imports for offloading and tiling
from ...utils import (
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import KandinskyImagePipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
        使用示例见文档

    Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

    Clean text using ftfy if available and unescape HTML entities.

    Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

    Normalize whitespace in text by replacing multiple spaces with single space.

    Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

    Apply both basic cleaning and whitespace normalization to prompts.

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds_qwen",
        "prompt_embeds_clip",
        "negative_prompt_embeds_qwen",
        "negative_prompt_embeds_clip",
    ]

    def __init__(
        self,
        transformer: Kandinsky5Transformer3DModel,
        vae: AutoencoderKL,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2VLProcessor,
        text_encoder_2: CLIPTextModel,
        tokenizer_2: CLIPTokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        )

        self.prompt_template = "<|im_start|>system\nYou are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
        self.prompt_template_encode_start_idx = 41

        self.vae_scale_factor_spatial = 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.resolutions = [(1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)]

    def _encode_prompt_qwen(
        self,
        prompt: List[str],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        dtype: Optional[torch.dtype] = None,
    ):

        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        full_texts = [self.prompt_template.format(p) for p in prompt]
        max_allowed_len = self.prompt_template_encode_start_idx + max_sequence_length

        untruncated_ids = self.tokenizer(
            text=full_texts,
            images=None,
            videos=None,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

        if untruncated_ids.shape[-1] > max_allowed_len:
            for i, text in enumerate(full_texts):
                tokens = untruncated_ids[i][self.prompt_template_encode_start_idx : -2]
                removed_text = self.tokenizer.decode(tokens[max_sequence_length - 2 :])
                if len(removed_text) > 0:
                    full_texts[i] = text[: -len(removed_text)]
                    logger.warning(
                        "The following part of your input was truncated because `max_sequence_length` is set to "
                        f" {max_sequence_length} tokens: {removed_text}"
                    )

        inputs = self.tokenizer(
            text=full_texts,
            images=None,
            videos=None,
            max_length=max_allowed_len,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(device)

        embeds = self.text_encoder(
            input_ids=inputs["input_ids"],
            return_dict=True,
            output_hidden_states=True,
        )["hidden_states"][-1][:, self.prompt_template_encode_start_idx :]
        attention_mask = inputs["attention_mask"][:, self.prompt_template_encode_start_idx :]
        cu_seqlens = torch.cumsum(attention_mask.sum(1), dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(dtype=torch.int32)

        return embeds.to(dtype), cu_seqlens

    def _encode_prompt_clip(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):

        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype

        inputs = self.tokenizer_2(
            prompt,
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        pooled_embed = self.text_encoder_2(**inputs)["pooler_output"]

        return pooled_embed.to(dtype)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        
        """r"""
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        if not isinstance(prompt, list):
            prompt = [prompt]

        batch_size = len(prompt)

        prompt = [prompt_clean(p) for p in prompt]

        # Encode with Qwen2.5-VL
        prompt_embeds_qwen, prompt_cu_seqlens = self._encode_prompt_qwen(
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            dtype=dtype,
        )
        # prompt_embeds_qwen shape: [batch_size, seq_len, embed_dim]

        # Encode with CLIP
        prompt_embeds_clip = self._encode_prompt_clip(
            prompt=prompt,
            device=device,
            dtype=dtype,
        )
        # prompt_embeds_clip shape: [batch_size, clip_embed_dim]

        # Repeat embeddings for num_images_per_prompt
        # Qwen embeddings: repeat sequence for each image, then reshape
        prompt_embeds_qwen = prompt_embeds_qwen.repeat(
            1, num_images_per_prompt, 1
        )  # [batch_size, seq_len * num_images_per_prompt, embed_dim]
        # Reshape to [batch_size * num_images_per_prompt, seq_len, embed_dim]
        prompt_embeds_qwen = prompt_embeds_qwen.view(
            batch_size * num_images_per_prompt, -1, prompt_embeds_qwen.shape[-1]
        )

        # CLIP embeddings: repeat for each image
        prompt_embeds_clip = prompt_embeds_clip.repeat(
            1, num_images_per_prompt, 1
        )  # [batch_size, num_images_per_prompt, clip_embed_dim]
        # Reshape to [batch_size * num_images_per_prompt, clip_embed_dim]
        prompt_embeds_clip = prompt_embeds_clip.view(batch_size * num_images_per_prompt, -1)

        # Repeat cumulative sequence lengths for num_images_per_prompt
        # Original differences (lengths) for each prompt in the batch
        original_lengths = prompt_cu_seqlens.diff()  # [len1, len2, ...]
        # Repeat the lengths for num_images_per_prompt
        repeated_lengths = original_lengths.repeat_interleave(
            num_images_per_prompt
        )  # [len1, len1, ..., len2, len2, ...]
        # Reconstruct the cumulative lengths
        repeated_cu_seqlens = torch.cat(
            [torch.tensor([0], device=device, dtype=torch.int32), repeated_lengths.cumsum(0)]
        )

        return prompt_embeds_qwen, prompt_embeds_clip, repeated_cu_seqlens

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds_qwen=None,
        prompt_embeds_clip=None,
        negative_prompt_embeds_qwen=None,
        negative_prompt_embeds_clip=None,
        prompt_cu_seqlens=None,
        negative_prompt_cu_seqlens=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):


        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError("max_sequence_length must be less than 1024")

        if (width, height) not in self.resolutions:
            resolutions_str = ",".join([f"({w},{h})" for w, h in self.resolutions])
            logger.warning(
                f"`height` and `width` have to be one of {resolutions_str}, but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # Check for consistency within positive prompt embeddings and sequence lengths
        if prompt_embeds_qwen is not None or prompt_embeds_clip is not None or prompt_cu_seqlens is not None:
            if prompt_embeds_qwen is None or prompt_embeds_clip is None or prompt_cu_seqlens is None:
                raise ValueError(
                    "If any of `prompt_embeds_qwen`, `prompt_embeds_clip`, or `prompt_cu_seqlens` is provided, "
                    "all three must be provided."
                )

        # Check for consistency within negative prompt embeddings and sequence lengths
        if (
            negative_prompt_embeds_qwen is not None
            or negative_prompt_embeds_clip is not None
            or negative_prompt_cu_seqlens is not None
        ):
            if (
                negative_prompt_embeds_qwen is None
                or negative_prompt_embeds_clip is None
                or negative_prompt_cu_seqlens is None
            ):
                raise ValueError(
                    "If any of `negative_prompt_embeds_qwen`, `negative_prompt_embeds_clip`, or `negative_prompt_cu_seqlens` is provided, "
                    "all three must be provided."
                )

        # Check if prompt or embeddings are provid...
        if prompt is None and prompt_embeds_qwen is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_qwen` (and corresponding `prompt_embeds_clip` and `prompt_cu_seqlens`). Cannot leave all undefined."
            )

        # Validate types for prompt and negative_prompt if provided
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 1024,
        width: int = 1024,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
            num_channels_latents,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Generate random noise
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):

        return self._guidance_scale

    @property
    def num_timesteps(self):

        return self._num_timesteps

    @property
    def interrupt(self):

        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds_qwen: Optional[torch.Tensor] = None,
        prompt_embeds_clip: Optional[torch.Tensor] = None,
        negative_prompt_embeds_qwen: Optional[torch.Tensor] = None,
        negative_prompt_embeds_clip: Optional[torch.Tensor] = None,
        prompt_cu_seqlens: Optional[torch.Tensor] = None,
        negative_prompt_cu_seqlens: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        
        """r"""
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds_qwen=prompt_embeds_qwen,
            prompt_embeds_clip=prompt_embeds_clip,
            negative_prompt_embeds_qwen=negative_prompt_embeds_qwen,
            negative_prompt_embeds_clip=negative_prompt_embeds_clip,
            prompt_cu_seqlens=prompt_cu_seqlens,
            negative_prompt_cu_seqlens=negative_prompt_cu_seqlens,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        if (width, height) not in self.resolutions:
            width, height = self.resolutions[
                np.argmin([abs((i[0] / i[1]) - (width / height)) for i in self.resolutions])
            ]

        self._guidance_scale = guidance_scale
        self._interrupt = False

        device = self._execution_device
        dtype = self.transformer.dtype

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds_qwen.shape[0]

        # 3. Encode input prompt
        if prompt_embeds_qwen is None:
            prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens = self.encode_prompt(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if self.guidance_scale > 1.0:
            if negative_prompt is None:
                negative_prompt = ""

            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt) if prompt is not None else [negative_prompt]
            elif len(negative_prompt) != len(prompt):
                raise ValueError(
                    f"`negative_prompt` must have same length as `prompt`. Got {len(negative_prompt)} vs {len(prompt)}."
                )

            if negative_prompt_embeds_qwen is None:
                negative_prompt_embeds_qwen, negative_prompt_embeds_clip, negative_prompt_cu_seqlens = (
                    self.encode_prompt(
                        prompt=negative_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        max_sequence_length=max_sequence_length,
                        device=device,
                        dtype=dtype,
                    )
                )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_visual_dim
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 6. Prepare rope positions for positional encoding
        visual_rope_pos = [
            torch.arange(1, device=device),
            torch.arange(height // self.vae_scale_factor_spatial // 2, device=device),
            torch.arange(width // self.vae_scale_factor_spatial // 2, device=device),
        ]

        text_rope_pos = torch.arange(prompt_cu_seqlens.diff().max().item(), device=device)

        negative_text_rope_pos = (
            torch.arange(negative_prompt_cu_seqlens.diff().max().item(), device=device)
            if negative_prompt_cu_seqlens is not None
            else None
        )

        # 7. Calculate dynamic scale factor based on resolution
        scale_factor = [1.0, 1.0, 1.0]

        # 8. Sparse Params for efficient attention
        sparse_params = None

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.unsqueeze(0).repeat(batch_size * num_images_per_prompt)

                # Predict noise residual
                pred_velocity = self.transformer(
                    hidden_states=latents.to(dtype),
                    encoder_hidden_states=prompt_embeds_qwen.to(dtype),
                    pooled_projections=prompt_embeds_clip.to(dtype),
                    timestep=timestep.to(dtype),
                    visual_rope_pos=visual_rope_pos,
                    text_rope_pos=text_rope_pos,
                    scale_factor=scale_factor,
                    sparse_params=sparse_params,
                    return_dict=True,
                ).sample

                if self.guidance_scale > 1.0 and negative_prompt_embeds_qwen is not None:
                    uncond_pred_velocity = self.transformer(
                        hidden_states=latents.to(dtype),
                        encoder_hidden_states=negative_prompt_embeds_qwen.to(dtype),
                        pooled_projections=negative_prompt_embeds_clip.to(dtype),
                        timestep=timestep.to(dtype),
                        visual_rope_pos=visual_rope_pos,
                        text_rope_pos=negative_text_rope_pos,
                        scale_factor=scale_factor,
                        sparse_params=sparse_params,
                        return_dict=True,
                    ).sample

                    pred_velocity = uncond_pred_velocity + guidance_scale * (pred_velocity - uncond_pred_velocity)

                latents = self.scheduler.step(pred_velocity[:, :], t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds_qwen = callback_outputs.pop("prompt_embeds_qwen", prompt_embeds_qwen)
                    prompt_embeds_clip = callback_outputs.pop("prompt_embeds_clip", prompt_embeds_clip)
                    negative_prompt_embeds_qwen = callback_outputs.pop(
                        "negative_prompt_embeds_qwen", negative_prompt_embeds_qwen
                    )
                    negative_prompt_embeds_clip = callback_outputs.pop(
                        "negative_prompt_embeds_clip", negative_prompt_embeds_clip
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 9. Post-processing - extract main latents
        latents = latents[:, :, :, :, :num_channels_latents]

        # 10. Decode latents to image
        if output_type != "latent":
            latents = latents.to(self.vae.dtype)
            # Reshape and normalize latents
            latents = latents.reshape(
                batch_size,
                num_images_per_prompt,
                1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
                num_channels_latents,
            )
            latents = latents.permute(0, 1, 5, 2, 3, 4)  # [batch, num_images, channels, 1, height, width]
            latents = latents.reshape(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )

            # Normalize and decode through VAE
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents).sample
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return KandinskyImagePipelineOutput(image=image)
