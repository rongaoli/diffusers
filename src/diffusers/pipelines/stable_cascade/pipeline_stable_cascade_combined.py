from typing import Callable, Dict, List, Optional, Union

import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from ...models import StableCascadeUNet
from ...schedulers import DDPMWuerstchenScheduler
from ...utils import is_torch_version, replace_example_docstring
from ..pipeline_utils import DeprecatedPipelineMixin, DiffusionPipeline
from ..wuerstchen.modeling_paella_vq_model import PaellaVQModel
from .pipeline_stable_cascade import StableCascadeDecoderPipeline
from .pipeline_stable_cascade_prior import StableCascadePriorPipeline

TEXT2IMAGE_EXAMPLE_DOC_STRING = """
        使用示例见文档


    _last_supported_version = "0.35.2"

    _load_connected_pipes = True
    _optional_components = ["prior_feature_extractor", "prior_image_encoder"]

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        decoder: StableCascadeUNet,
        scheduler: DDPMWuerstchenScheduler,
        vqgan: PaellaVQModel,
        prior_prior: StableCascadeUNet,
        prior_text_encoder: CLIPTextModelWithProjection,
        prior_tokenizer: CLIPTokenizer,
        prior_scheduler: DDPMWuerstchenScheduler,
        prior_feature_extractor: Optional[CLIPImageProcessor] = None,
        prior_image_encoder: Optional[CLIPVisionModelWithProjection] = None,
    ):
        super().__init__()

        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            decoder=decoder,
            scheduler=scheduler,
            vqgan=vqgan,
            prior_text_encoder=prior_text_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_prior=prior_prior,
            prior_scheduler=prior_scheduler,
            prior_feature_extractor=prior_feature_extractor,
            prior_image_encoder=prior_image_encoder,
        )
        self.prior_pipe = StableCascadePriorPipeline(
            prior=prior_prior,
            text_encoder=prior_text_encoder,
            tokenizer=prior_tokenizer,
            scheduler=prior_scheduler,
            image_encoder=prior_image_encoder,
            feature_extractor=prior_feature_extractor,
        )
        self.decoder_pipe = StableCascadeDecoderPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            decoder=decoder,
            scheduler=scheduler,
            vqgan=vqgan,
        )

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = None):
        
        """r"""
        self.prior_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.decoder_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = None):
        
        """r"""
        self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
        self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)

    def progress_bar(self, iterable=None, total=None):
        self.prior_pipe.progress_bar(iterable=iterable, total=total)
        self.decoder_pipe.progress_bar(iterable=iterable, total=total)

    def set_progress_bar_config(self, **kwargs):
        self.prior_pipe.set_progress_bar_config(**kwargs)
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    @replace_example_docstring(TEXT2IMAGE_EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[str] = None,
        images: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]] = None,
        height: int = 512,
        width: int = 512,
        prior_num_inference_steps: int = 60,
        prior_guidance_scale: float = 4.0,
        num_inference_steps: int = 12,
        decoder_guidance_scale: float = 0.0,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_pooled: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_pooled: Optional[torch.Tensor] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        prior_callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        prior_callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        Function invoked when calling the pipeline for generation.
                `prior_timesteps`
            num_inference_steps (`int`, *optional*, defaults to 12):
                The number of decoder denoising steps. More denoising steps usually lead to a higher quality image at
                the expense of slower inference. For more specific timestep spacing, you can pass customized
                `timesteps`
            decoder_guidance_scale (`float`, *optional*, defaults to 0.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            prior_callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `prior_callback_on_step_end(self: DiffusionPipeline, step: int, timestep:
                int, callback_kwargs: Dict)`.
            prior_callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `prior_callback_on_step_end` function. The tensors specified in the
                list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in
                the `._callback_tensor_inputs` attribute of your pipeline class.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:
