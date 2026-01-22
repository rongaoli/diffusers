import math
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ...models import PriorTransformer
from ...schedulers import HeunDiscreteScheduler
from ...utils import (
    BaseOutput,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .renderer import ShapERenderer

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
        使用示例见文档


    images: Union[List[List[PIL.Image.Image]], List[List[np.ndarray]]]

class ShapEPipeline(DiffusionPipeline):


    model_cpu_offload_seq = "text_encoder->prior"
    _exclude_from_cpu_offload = ["shap_e_renderer"]

    def __init__(
        self,
        prior: PriorTransformer,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        scheduler: HeunDiscreteScheduler,
        shap_e_renderer: ShapERenderer,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            shap_e_renderer=shap_e_renderer,
        )

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
    ):
        len(prompt) if isinstance(prompt, list) else 1

        # YiYi Notes: set pad_token_id to be 0, not sure why I can't set in the config file
        self.tokenizer.pad_token_id = 0
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        text_encoder_output = self.text_encoder(text_input_ids.to(device))
        prompt_embeds = text_encoder_output.text_embeds

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        # in Shap-E it normalize the prompt_embeds and then later rescale it
        prompt_embeds = prompt_embeds / torch.linalg.norm(prompt_embeds, dim=-1, keepdim=True)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Rescale the features to have unit variance
        prompt_embeds = math.sqrt(prompt_embeds.shape[1]) * prompt_embeds

        return prompt_embeds

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        guidance_scale: float = 4.0,
        frame_size: int = 64,
        output_type: Optional[str] = "pil",  # pil, np, latent, mesh
        return_dict: bool = True,
    ):
        The call function to the pipeline for generation.
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            frame_size (`int`, *optional*, default to 64):
                The width and height of each image frame of the generated 3D output.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`), `"latent"` (`torch.Tensor`), or mesh ([`MeshDecoderOutput`]).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput`] instead of a plain
                tuple.

        Examples:
