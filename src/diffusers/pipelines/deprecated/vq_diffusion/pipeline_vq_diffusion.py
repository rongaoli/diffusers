from typing import Callable, List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from ....configuration_utils import ConfigMixin, register_to_config
from ....models import ModelMixin, Transformer2DModel, VQModel
from ....schedulers import VQDiffusionScheduler
from ....utils import logging
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class LearnedClassifierFreeSamplingEmbeddings(ModelMixin, ConfigMixin):


    @register_to_config
    def __init__(self, learnable: bool, hidden_size: Optional[int] = None, length: Optional[int] = None):
        super().__init__()

        self.learnable = learnable

        if self.learnable:
            assert hidden_size is not None, "learnable=True requires `hidden_size` to be set"
            assert length is not None, "learnable=True requires `length` to be set"

            embeddings = torch.zeros(length, hidden_size)
        else:
            embeddings = None

        self.embeddings = torch.nn.Parameter(embeddings)

class VQDiffusionPipeline(DiffusionPipeline):


    vqvae: VQModel
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    transformer: Transformer2DModel
    learned_classifier_free_sampling_embeddings: LearnedClassifierFreeSamplingEmbeddings
    scheduler: VQDiffusionScheduler

    def __init__(
        self,
        vqvae: VQModel,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        transformer: Transformer2DModel,
        scheduler: VQDiffusionScheduler,
        learned_classifier_free_sampling_embeddings: LearnedClassifierFreeSamplingEmbeddings,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            learned_classifier_free_sampling_embeddings=learned_classifier_free_sampling_embeddings,
        )

    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        prompt_embeds = self.text_encoder(text_input_ids.to(self.device))[0]

        # NOTE: This additional step of normalizing the text embeddings is from VQ-Diffusion.
        # While CLIP does normalize the pooled output of the text transformer when combining
        # the image and text embeddings, CLIP does not directly normalize the last hidden state.
        #
        # CLIP normalizing the pooled output.
        # https://github.com/huggingface/transform...
        prompt_embeds = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)

        # duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            if self.learned_classifier_free_sampling_embeddings.learnable:
                negative_prompt_embeds = self.learned_classifier_free_sampling_embeddings.embeddings
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                uncond_tokens = [""] * batch_size

                max_length = text_input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
                # See comment for normalizing text embeddings
                negative_prompt_embeds = negative_prompt_embeds / negative_prompt_embeds.norm(dim=-1, keepdim=True)

            # duplicate unconditional embeddings f...
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 100,
        guidance_scale: float = 5.0,
        truncation_rate: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> Union[ImagePipelineOutput, Tuple]:
        The call function to the pipeline for generation.
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            truncation_rate (`float`, *optional*, defaults to 1.0 (equivalent to no truncation)):
                Used to "truncate" the predicted classes for x_0 such that the cumulative probability for a pixel is at
                most `truncation_rate`. The lowest probabilities that would increase the cumulative probability above
                `truncation_rate` are set to zero.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor` of shape (batch), *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Must be valid embedding indices.If not provided, a latents tensor will be generated of
                completely masked latent pixels.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
