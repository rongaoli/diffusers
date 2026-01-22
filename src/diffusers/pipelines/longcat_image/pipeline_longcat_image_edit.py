import inspect
import math
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from ...image_processor import VaeImageProcessor
from ...loaders import FromSingleFileMixin
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import LongCatImageTransformer2DModel
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from .pipeline_output import LongCatImagePipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
        使用示例见文档

    Implement a regex-based string splitting algorithm that identifies delimiters defined by single or double quote
    pairs. Examples::
        >>> prompt_en = "Please write 'Hello' on the blackboard for me." >>> print(split_quotation(prompt_en)) >>> #
        output: [('Please write ', False), ("'Hello'", True), (' on the blackboard for me.', False)]

    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.
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

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = width if width % 16 == 0 else (width // 16 + 1) * 16
    height = height if height % 16 == 0 else (height // 16 + 1) * 16

    width = int(width)
    height = int(height)

    return width, height

class LongCatImageEditPipeline(DiffusionPipeline, FromSingleFileMixin):


    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        text_processor: Qwen2VLProcessor,
        transformer: LongCatImageTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            text_processor=text_processor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.image_processor_vl = text_processor.image_processor

        self.image_token = "<|image_pad|>"
        self.prompt_template_encode_prefix = "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.default_sample_size = 128
        self.tokenizer_max_length = 512

    def _encode_prompt(self, prompt, image):
        raw_vl_input = self.image_processor_vl(images=image, return_tensors="pt")
        pixel_values = raw_vl_input["pixel_values"]
        image_grid_thw = raw_vl_input["image_grid_thw"]
        all_tokens = []
        for clean_prompt_sub, matched in split_quotation(prompt[0]):
            if matched:
                for sub_word in clean_prompt_sub:
                    tokens = self.tokenizer(sub_word, add_special_tokens=False)["input_ids"]
                    all_tokens.extend(tokens)
            else:
                tokens = self.tokenizer(clean_prompt_sub, add_special_tokens=False)["input_ids"]
                all_tokens.extend(tokens)

        if len(all_tokens) > self.tokenizer_max_length:
            logger.warning(
                "Your input was truncated because `max_sequence_length` is set to "
                f" {self.tokenizer_max_length} input token nums : {len(len(all_tokens))}"
            )
            all_tokens = all_tokens[: self.tokenizer_max_length]

        text_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": [all_tokens]},
            max_length=self.tokenizer_max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        text = self.prompt_template_encode_prefix

        merge_length = self.image_processor_vl.merge_size**2
        while self.image_token in text:
            num_image_tokens = image_grid_thw.prod() // merge_length
            text = text.replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
        text = text.replace("<|placeholder|>", self.image_token)

        prefix_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        suffix_tokens = self.tokenizer(self.prompt_template_encode_suffix, add_special_tokens=False)["input_ids"]

        vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        prefix_len = prefix_tokens.index(vision_start_token_id)
        suffix_len = len(suffix_tokens)

        prefix_tokens_mask = torch.tensor([1] * len(prefix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)
        suffix_tokens_mask = torch.tensor([1] * len(suffix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)

        prefix_tokens = torch.tensor(prefix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)
        suffix_tokens = torch.tensor(suffix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)

        input_ids = torch.cat((prefix_tokens, text_tokens_and_mask.input_ids[0], suffix_tokens), dim=-1)
        attention_mask = torch.cat(
            (prefix_tokens_mask, text_tokens_and_mask.attention_mask[0], suffix_tokens_mask), dim=-1
        )

        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)

        pixel_values = pixel_values.to(self.device)
        image_grid_thw = image_grid_thw.to(self.device)

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        # [max_sequence_length, batch, hidden_size] -> [batch, max_sequence_length, hidden_size]
        # clone to have a contiguous tensor
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[:, prefix_len:-suffix_len, :]
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: List[str] = None,
        image: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        # If prompt_embeds is provided and prompt is None, skip encoding
        if prompt_embeds is None:
            prompt_embeds = self._encode_prompt(prompt, image)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention...
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=prompt_embeds.shape[1]).to(
            self.device
        )
        return prompt_embeds, text_ids

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        prompt_embeds_length,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        image_latents, image_latents_ids = None, None

        if image is not None:
            image = image.to(device=self.device, dtype=dtype)

            if image.shape[1] != self.vae.config.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)

            image_latents_ids = prepare_pos_ids(
                modality_id=2,
                type="image",
                start=(prompt_embeds_length, prompt_embeds_length),
                height=height // 2,
                width=width // 2,
            ).to(device, dtype=torch.float64)

        shape = (batch_size, num_channels_latents, height, width)
        latents_ids = prepare_pos_ids(
            modality_id=1,
            type="image",
            start=(prompt_embeds_length, prompt_embeds_length),
            height=height // 2,
            width=width // 2,
        ).to(device)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents, latents_ids, image_latents_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def check_inputs(
        self, prompt, height, width, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None:
            if isinstance(prompt, str):
                pass
            elif isinstance(prompt, list) and len(prompt) == 1:
                pass
            else:
                raise ValueError(
                    f"`prompt` must be a `str` or a `list` of length 1, but is {prompt} (type: {type(prompt)})"
                )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PIL.Image.Image] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        Function invoked when calling the pipeline for generation.

        Examples:
