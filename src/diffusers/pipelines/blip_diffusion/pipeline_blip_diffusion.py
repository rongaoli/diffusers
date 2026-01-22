from typing import List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPTokenizer

from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import PNDMScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DeprecatedPipelineMixin, DiffusionPipeline, ImagePipelineOutput
from .blip_image_processing import BlipImageProcessor
from .modeling_blip2 import Blip2QFormerModel
from .modeling_ctx_clip import ContextCLIPTextModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """


class BlipDiffusionPipeline(DeprecatedPipelineMixin, DiffusionPipeline):


    _last_supported_version = "0.33.1"
    model_cpu_offload_seq = "qformer->text_encoder->unet->vae"

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: ContextCLIPTextModel,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        qformer: Blip2QFormerModel,
        image_processor: BlipImageProcessor,
        ctx_begin_pos: int = 2,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            qformer=qformer,
            image_processor=image_processor,
        )
        self.register_to_config(ctx_begin_pos=ctx_begin_pos, mean=mean, std=std)

    def get_query_embeddings(self, input_image, src_subject):
        return self.qformer(image_input=input_image, text_input=src_subject, return_dict=False)

    # from the original Blip Diffusion code, speci...
    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0, prompt_reps=20):
        rv = []
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # a trick to amplify the prompt
            rv.append(", ".join([prompt] * int(prompt_strength * prompt_reps)))

        return rv

    # Copied from diffusers.pipelines.consistency_...
    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def encode_prompt(self, query_embeds, prompt, device=None):
        device = device or self._execution_device

        # embeddings for prompt, with query_embeds as context
        max_len = self.text_encoder.text_model.config.max_position_embeddings
        max_len -= self.qformer.config.num_query_tokens

        tokenized_prompt = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        batch_size = query_embeds.shape[0]
        ctx_begin_pos = [self.config.ctx_begin_pos] * batch_size

        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=ctx_begin_pos,
        )[0]

        return text_embeddings

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: List[str],
        reference_image: PIL.Image.Image,
        source_subject_category: List[str],
        target_subject_category: List[str],
        latents: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        neg_prompt: Optional[str] = "",
        prompt_strength: float = 1.0,
        prompt_reps: int = 20,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        Function invoked when calling the pipeline for generation.
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            height (`int`, *optional*, defaults to 512):
                The height of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            neg_prompt (`str`, *optional*, defaults to ""):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_strength (`float`, *optional*, defaults to 1.0):
                The strength of the prompt. Specifies the number of times the prompt is repeated along with prompt_reps
                to amplify the prompt.
            prompt_reps (`int`, *optional*, defaults to 20):
                The number of times the prompt is repeated along with prompt_strength to amplify the prompt.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        Examples:
