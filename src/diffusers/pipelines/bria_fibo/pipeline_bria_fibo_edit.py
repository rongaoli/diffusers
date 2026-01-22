import json
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.smollm3.modeling_smollm3 import SmolLM3ForCausalLM

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FluxLoraLoaderMixin
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel
from ...pipelines.bria_fibo.pipeline_output import BriaFiboPipelineOutput
from ...pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler, KarrasDiffusionSchedulers
from ...utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

PipelineMaskInput = Union[
    torch.FloatTensor, Image.Image, List[Image.Image], List[torch.FloatTensor], np.ndarray, List[np.ndarray]
]

# TODO: Update example docstring
EXAMPLE_DOC_STRING = """


PREFERRED_RESOLUTION = {
    256 * 256: [(208, 304), (224, 288), (256, 256), (288, 224), (304, 208), (320, 192), (336, 192)],
    512 * 512: [
        (416, 624),
        (432, 592),
        (464, 560),
        (512, 512),
        (544, 480),
        (576, 448),
        (592, 432),
        (608, 416),
        (624, 416),
        (640, 400),
        (672, 384),
        (704, 368),
    ],
    1024 * 1024: [
        (832, 1248),
        (880, 1184),
        (912, 1136),
        (1024, 1024),
        (1136, 912),
        (1184, 880),
        (1216, 848),
        (1248, 832),
        (1248, 832),
        (1264, 816),
        (1296, 800),
        (1360, 768),
    ],
}

def is_valid_edit_json(json_input: str | dict):

    try:
        if isinstance(json_input, str) and "edit_instruction" in json_input:
            json.loads(json_input)
            return True
        elif isinstance(json_input, dict) and "edit_instruction" in json_input:
            return True
        else:
            return False
    except json.JSONDecodeError:
        return False

def is_valid_mask(mask: PipelineMaskInput):

    if isinstance(mask, torch.Tensor):
        return True
    elif isinstance(mask, Image.Image):
        return True
    elif isinstance(mask, list):
        return all(isinstance(m, (torch.Tensor, Image.Image, np.ndarray)) for m in mask)
    elif isinstance(mask, np.ndarray):
        return mask.ndim in [2, 3] and mask.min() >= 0 and mask.max() <= 1
    else:
        return False

def get_mask_size(mask: PipelineMaskInput):

    if isinstance(mask, torch.Tensor):
        return mask.shape[-2:]
    elif isinstance(mask, Image.Image):
        return mask.size[::-1]  # (height, width)
    elif isinstance(mask, list):
        return [get_mask_size(m) for m in mask]
    elif isinstance(mask, np.ndarray):
        return mask.shape[-2:]
    else:
        return None

def get_image_size(image: PipelineImageInput):

    if isinstance(image, torch.Tensor):
        return image.shape[-2:]
    elif isinstance(image, Image.Image):
        return image.size[::-1]  # (height, width)
    elif isinstance(image, list):
        return [get_image_size(i) for i in image]
    else:
        return None

def paste_mask_on_image(mask: PipelineMaskInput, image: PipelineImageInput):

    if isinstance(mask, torch.Tensor):
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        mask = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
    elif isinstance(mask, Image.Image):
        pass
    elif isinstance(mask, list):
        mask = mask[0]
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            mask = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
        elif isinstance(mask, np.ndarray):
            mask = Image.fromarray((mask * 255).astype(np.uint8))
    elif isinstance(mask, np.ndarray):
        mask = Image.fromarray((mask * 255).astype(np.uint8))

    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
    elif isinstance(image, Image.Image):
        pass
    elif isinstance(image, list):
        image = image[0]
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.permute(1, 2, 0)
            image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))

    mask = mask.convert("L")
    image = image.convert("RGB")
    gray_color = (128, 128, 128)
    gray_img = Image.new("RGB", image.size, gray_color)
    image = Image.composite(gray_img, image, mask)
    return image

class BriaFiboEditPipeline(DiffusionPipeline, FluxLoraLoaderMixin):


    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        transformer: BriaFiboTransformer2DModel,
        scheduler: Union[FlowMatchEulerDiscreteScheduler, KarrasDiffusionSchedulers],
        vae: AutoencoderKLWan,
        text_encoder: SmolLM3ForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)  # * 2)
        self.default_sample_size = 32  # 64

    def get_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if not prompt:
            raise ValueError("`prompt` must be a non-empty string or list of strings.")

        batch_size = len(prompt)
        bot_token_id = 128000

        text_encoder_device = device if device is not None else torch.device("cpu")
        if not isinstance(text_encoder_device, torch.device):
            text_encoder_device = torch.device(text_encoder_device)

        if all(p == "" for p in prompt):
            input_ids = torch.full((batch_size, 1), bot_token_id, dtype=torch.long, device=text_encoder_device)
            attention_mask = torch.ones_like(input_ids)
        else:
            tokenized = self.tokenizer(
                prompt,
                padding="longest",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(text_encoder_device)
            attention_mask = tokenized.attention_mask.to(text_encoder_device)

            if any(p == "" for p in prompt):
                empty_rows = torch.tensor([p == "" for p in prompt], dtype=torch.bool, device=text_encoder_device)
                input_ids[empty_rows] = bot_token_id
                attention_mask[empty_rows] = 1

        encoder_outputs = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_outputs.hidden_states

        prompt_embeds = torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        hidden_states = tuple(
            layer.repeat_interleave(num_images_per_prompt, dim=0).to(device=device) for layer in hidden_states
        )
        attention_mask = attention_mask.repeat_interleave(num_images_per_prompt, dim=0).to(device=device)

        return prompt_embeds, hidden_states, attention_mask

    @staticmethod
    def pad_embedding(prompt_embeds, max_tokens, attention_mask=None):
        # Pad embeddings to `max_tokens` while preserving the mask of real tokens.
        batch_size, seq_len, dim = prompt_embeds.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=prompt_embeds.dtype, device=prompt_embeds.device)
        else:
            attention_mask = attention_mask.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)

        if max_tokens < seq_len:
            raise ValueError("`max_tokens` must be greater or equal to the current sequence length.")

        if max_tokens > seq_len:
            pad_length = max_tokens - seq_len
            padding = torch.zeros(
                (batch_size, pad_length, dim), dtype=prompt_embeds.dtype, device=prompt_embeds.device
            )
            prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)

            mask_padding = torch.zeros(
                (batch_size, pad_length), dtype=prompt_embeds.dtype, device=prompt_embeds.device
            )
            attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

        return prompt_embeds, attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 5,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 3000,
        lora_scale: Optional[float] = None,
    ):
        
        """r"""
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_attention_mask = None
        negative_prompt_attention_mask = None
        if prompt_embeds is None:
            prompt_embeds, prompt_layers, prompt_attention_mask = self.get_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_embeds = prompt_embeds.to(dtype=self.transformer.dtype)
            prompt_layers = [tensor.to(dtype=self.transformer.dtype) for tensor in prompt_layers]

        if guidance_scale > 1:
            if isinstance(negative_prompt, list) and negative_prompt[0] is None:
                negative_prompt = ""
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

            negative_prompt_embeds, negative_prompt_layers, negative_prompt_attention_mask = self.get_prompt_embeds(
                prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.transformer.dtype)
            negative_prompt_layers = [tensor.to(dtype=self.transformer.dtype) for tensor in negative_prompt_layers]

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        # Pad to longest
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)

        if negative_prompt_embeds is not None:
            if negative_prompt_attention_mask is not None:
                negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                    device=negative_prompt_embeds.device, dtype=negative_prompt_embeds.dtype
                )
            max_tokens = max(negative_prompt_embeds.shape[1], prompt_embeds.shape[1])

            prompt_embeds, prompt_attention_mask = self.pad_embedding(
                prompt_embeds, max_tokens, attention_mask=prompt_attention_mask
            )
            prompt_layers = [self.pad_embedding(layer, max_tokens)[0] for layer in prompt_layers]

            negative_prompt_embeds, negative_prompt_attention_mask = self.pad_embedding(
                negative_prompt_embeds, max_tokens, attention_mask=negative_prompt_attention_mask
            )
            negative_prompt_layers = [self.pad_embedding(layer, max_tokens)[0] for layer in negative_prompt_layers]
        else:
            max_tokens = prompt_embeds.shape[1]
            prompt_embeds, prompt_attention_mask = self.pad_embedding(
                prompt_embeds, max_tokens, attention_mask=prompt_attention_mask
            )
            negative_prompt_layers = None

        dtype = self.text_encoder.dtype
        text_ids = torch.zeros(prompt_embeds.shape[0], max_tokens, 3).to(device=device, dtype=dtype)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            text_ids,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_layers,
            negative_prompt_layers,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @staticmethod
    # Based on diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _unpack_latents_no_patch(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels)
        latents = latents.permute(0, 3, 1, 2)

        return latents

    @staticmethod
    def _pack_latents_no_patch(latents, batch_size, num_channels_latents, height, width):
        latents = latents.permute(0, 2, 3, 1)
        latents = latents.reshape(batch_size, height * width, num_channels_latents)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        do_patching=False,
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if do_patching:
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        else:
            latents = self._pack_latents_no_patch(latents, batch_size, num_channels_latents, height, width)
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        return latents, latent_image_ids

    @staticmethod
    def _prepare_attention_mask(attention_mask):
        attention_matrix = torch.einsum("bi,bj->bij", attention_mask, attention_mask)

        # convert to 0 - keep, -inf ignore
        attention_matrix = torch.where(
            attention_matrix == 1, 0.0, -torch.inf
        )  # Apply -inf to ignored tokens for nulling softmax score
        return attention_matrix

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[PipelineImageInput] = None,
        mask: Optional[PipelineMaskInput] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        timesteps: List[int] = None,
        seed: Optional[int] = None,
        guidance_scale: float = 5,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 3000,
        do_patching=False,
        _auto_resize: bool = True,
    ):

        Function invoked when calling the pipeline for generation.
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 3000): Maximum sequence length to use with the `prompt`.
            do_patching (`bool`, *optional*, defaults to `False`): Whether to use patching.
        Examples:
