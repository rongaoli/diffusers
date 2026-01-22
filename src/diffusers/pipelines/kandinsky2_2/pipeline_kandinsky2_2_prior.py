from typing import Callable, Dict, List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from ...models import PriorTransformer
from ...schedulers import UnCLIPScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..kandinsky import KandinskyPriorPipelineOutput
from ..pipeline_utils import DiffusionPipeline

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
        使用示例见文档

class KandinskyV22PriorPipeline(DiffusionPipeline):


    model_cpu_offload_seq = "text_encoder->image_encoder->prior"
    _exclude_from_cpu_offload = ["prior"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "text_encoder_hidden_states", "text_mask"]

    def __init__(
        self,
        prior: PriorTransformer,
        image_encoder: CLIPVisionModelWithProjection,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        scheduler: UnCLIPScheduler,
        image_processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
        )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_INTERPOLATE_DOC_STRING)
    def interpolate(
        self,
        images_and_prompts: List[Union[str, PIL.Image.Image, torch.Tensor]],
        weights: List[float],
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        negative_prior_prompt: Optional[str] = None,
        negative_prompt: str = "",
        guidance_scale: float = 4.0,
        device=None,
    ):
        Function invoked when using the prior pipeline for interpolation.
                `guidance_scale` is less than `1`).
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.

        Examples:
