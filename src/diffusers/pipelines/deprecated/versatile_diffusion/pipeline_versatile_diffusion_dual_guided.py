import inspect
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.utils.checkpoint
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ....image_processor import VaeImageProcessor
from ....models import AutoencoderKL, DualTransformer2DModel, Transformer2DModel, UNet2DConditionModel
from ....schedulers import KarrasDiffusionSchedulers
from ....utils import deprecate, logging
from ....utils.torch_utils import randn_tensor
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .modeling_text_unet import UNetFlatConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class VersatileDiffusionDualGuidedPipeline(DiffusionPipeline):


    model_cpu_offload_seq = "bert->unet->vqvae"

    tokenizer: CLIPTokenizer
    image_feature_extractor: CLIPImageProcessor
    text_encoder: CLIPTextModelWithProjection
    image_encoder: CLIPVisionModelWithProjection
    image_unet: UNet2DConditionModel
    text_unet: UNetFlatConditionModel
    vae: AutoencoderKL
    scheduler: KarrasDiffusionSchedulers

    _optional_components = ["text_unet"]

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        image_feature_extractor: CLIPImageProcessor,
        text_encoder: CLIPTextModelWithProjection,
        image_encoder: CLIPVisionModelWithProjection,
        image_unet: UNet2DConditionModel,
        text_unet: UNetFlatConditionModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            image_feature_extractor=image_feature_extractor,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            image_unet=image_unet,
            text_unet=text_unet,
            vae=vae,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if self.text_unet is not None and (
            "dual_cross_attention" not in self.image_unet.config or not self.image_unet.config.dual_cross_attention
        ):
            # if loading from a universal checkpoint rather than a saved dual-guided pipeline
            self._convert_to_dual_attention()

    def remove_unused_weights(self):
        self.register_modules(text_unet=None)

    def _convert_to_dual_attention(self):

        for name, module in self.image_unet.named_modules():
            if isinstance(module, Transformer2DModel):
                parent_name, index = name.rsplit(".", 1)
                index = int(index)

                image_transformer = self.image_unet.get_submodule(parent_name)[index]
                text_transformer = self.text_unet.get_submodule(parent_name)[index]

                config = image_transformer.config
                dual_transformer = DualTransformer2DModel(
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                    in_channels=config.in_channels,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    norm_num_groups=config.norm_num_groups,
                    cross_attention_dim=config.cross_attention_dim,
                    attention_bias=config.attention_bias,
                    sample_size=config.sample_size,
                    num_vector_embeds=config.num_vector_embeds,
                    activation_fn=config.activation_fn,
                    num_embeds_ada_norm=config.num_embeds_ada_norm,
                )
                dual_transformer.transformers[0] = image_transformer
                dual_transformer.transformers[1] = text_transformer

                self.image_unet.get_submodule(parent_name)[index] = dual_transformer
                self.image_unet.register_to_config(dual_cross_attention=True)

    def _revert_dual_attention(self):

        for name, module in self.image_unet.named_modules():
            if isinstance(module, DualTransformer2DModel):
                parent_name, index = name.rsplit(".", 1)
                index = int(index)
                self.image_unet.get_submodule(parent_name)[index] = module.transformers[0]

        self.image_unet.register_to_config(dual_cross_attention=False)

    def _encode_text_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance):
        
        """r"""

        def normalize_embeddings(encoder_output):
            embeds = self.text_encoder.text_projection(encoder_output.last_hidden_state)
            embeds_pooled = encoder_output.text_embeds
            embeds = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1, keepdim=True)
            return embeds

        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="pt").input_ids

        if not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = normalize_embeddings(prompt_embeds)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size
            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = normalize_embeddings(negative_prompt_embeds)

            # duplicate unconditional embeddings f...
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def _encode_image_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance):
        
        """r"""

        def normalize_embeddings(encoder_output):
            embeds = self.image_encoder.vision_model.post_layernorm(encoder_output.last_hidden_state)
            embeds = self.image_encoder.visual_projection(embeds)
            embeds_pooled = embeds[:, 0:1]
            embeds = embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)
            return embeds

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        image_input = self.image_feature_extractor(images=prompt, return_tensors="pt")
        pixel_values = image_input.pixel_values.to(device).to(self.image_encoder.dtype)
        image_embeddings = self.image_encoder(pixel_values)
        image_embeddings = normalize_embeddings(image_embeddings)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_images = [np.zeros((512, 512, 3)) + 0.5] * batch_size
            uncond_images = self.image_feature_extractor(images=uncond_images, return_tensors="pt")
            pixel_values = uncond_images.pixel_values.to(device).to(self.image_encoder.dtype)
            negative_prompt_embeds = self.image_encoder(pixel_values)
            negative_prompt_embeds = normalize_embeddings(negative_prompt_embeds)

            # duplicate unconditional embeddings f...
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and conditional embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings

    # Copied from diffusers.pipelines.stable_diffu...
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does n...
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffu...
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler s...
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, image, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, PIL.Image.Image) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` `PIL.Image` or `list` but is {type(prompt)}")
        if not isinstance(image, str) and not isinstance(image, PIL.Image.Image) and not isinstance(image, list):
            raise ValueError(f"`image` has to be of type `str` `PIL.Image` or `list` but is {type(image)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # Copied from diffusers.pipelines.stable_diffu...
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def set_transformer_params(self, mix_ratio: float = 0.5, condition_types: Tuple = ("text", "image")):
        for name, module in self.image_unet.named_modules():
            if isinstance(module, DualTransformer2DModel):
                module.mix_ratio = mix_ratio

                for i, type in enumerate(condition_types):
                    if type == "text":
                        module.condition_lengths[i] = self.text_encoder.config.max_position_embeddings
                        module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                    else:
                        module.condition_lengths[i] = 257
                        module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[PIL.Image.Image, List[PIL.Image.Image]],
        image: Union[str, List[str]],
        text_to_image_strength: float = 0.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        **kwargs,
    ):
        
        """r"""

        使用示例见文档

