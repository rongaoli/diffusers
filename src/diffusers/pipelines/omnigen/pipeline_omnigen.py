import inspect
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import LlamaTokenizer

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import OmniGenTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import deprecate, is_torch_xla_available, is_torchvision_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

if is_torchvision_available():
    from .processor_omnigen import OmniGenMultiModalProcessor

if is_torch_xla_available():
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
        使用示例见文档

    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.
):
    
    """r"""

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: OmniGenTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        tokenizer: LlamaTokenizer,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) is not None else 8
        )
        # OmniGen latents are turned into 2x2 patc...
        # by the patch size. So the vae scale fact...
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        self.multimodal_processor = OmniGenMultiModalProcessor(tokenizer, max_image_size=1024)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 120000
        )
        self.default_sample_size = 128

    def encode_input_images(
        self,
        input_pixel_values: List[torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):

        device = device or self._execution_device
        dtype = dtype or self.vae.dtype

        input_img_latents = []
        for img in input_pixel_values:
            img = self.vae.encode(img.to(device, dtype)).latent_dist.sample().mul_(self.vae.config.scaling_factor)
            input_img_latents.append(img)
        return input_img_latents

    def check_inputs(
        self,
        prompt,
        input_images,
        height,
        width,
        use_input_image_size_as_output,
        callback_on_step_end_tensor_inputs=None,
    ):
        if input_images is not None:
            if len(input_images) != len(prompt):
                raise ValueError(
                    f"The number of prompts: {len(prompt)} does not match the number of input images: {len(input_images)}."
                )
            for i in range(len(input_images)):
                if input_images[i] is not None:
                    if not all(f"<img><|image_{k + 1}|></img>" in prompt[i] for k in range(len(input_images[i]))):
                        raise ValueError(
                            f"prompt `{prompt[i]}` doesn't have enough placeholders for the input images `{input_images[i]}`"
                        )

        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if use_input_image_size_as_output:
            if input_images is None or input_images[0] is None:
                raise ValueError(
                    "`use_input_image_size_as_output` is set to True, but no input image was found. If you are performing a text-to-image task, please set it to False."
                )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    def enable_vae_slicing(self):
        
        """r"""
        depr_message = f"Calling `enable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_slicing()`."
        deprecate(
            "enable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        
        """r"""
        depr_message = f"Calling `disable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_slicing()`."
        deprecate(
            "disable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        
        """r"""
        depr_message = f"Calling `enable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_tiling()`."
        deprecate(
            "enable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        
        """r"""
        depr_message = f"Calling `disable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_tiling()`."
        deprecate(
            "disable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_tiling()

    # Copied from diffusers.pipelines.stable_diffu...
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
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

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
        prompt: Union[str, List[str]],
        input_images: Union[PipelineImageInput, List[PipelineImageInput]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        max_input_image_size: int = 1024,
        timesteps: List[int] = None,
        guidance_scale: float = 2.5,
        img_guidance_scale: float = 1.6,
        use_input_image_size_as_output: bool = False,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        
        """r"""

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        num_cfg = 2 if input_images is not None else 1
        use_img_cfg = True if input_images is not None else False
        if isinstance(prompt, str):
            prompt = [prompt]
            input_images = [input_images]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            input_images,
            height,
            width,
            use_input_image_size_as_output,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Define call parameters
        batch_size = len(prompt)
        device = self._execution_device

        # 3. process multi-modal instructions
        if max_input_image_size != self.multimodal_processor.max_image_size:
            self.multimodal_processor.reset_max_image_size(max_image_size=max_input_image_size)
        processed_data = self.multimodal_processor(
            prompt,
            input_images,
            height=height,
            width=width,
            use_img_cfg=use_img_cfg,
            use_input_image_size_as_output=use_input_image_size_as_output,
            num_images_per_prompt=num_images_per_prompt,
        )
        processed_data["input_ids"] = processed_data["input_ids"].to(device)
        processed_data["attention_mask"] = processed_data["attention_mask"].to(device)
        processed_data["position_ids"] = processed_data["position_ids"].to(device)

        # 4. Encode input images
        input_img_latents = self.encode_input_images(processed_data["input_pixel_values"], device=device)

        # 5. Prepare timesteps
        sigmas = np.linspace(1, 0, num_inference_steps + 1)[:num_inference_steps]
        if XLA_AVAILABLE:
            timestep_device = "cpu"
        else:
            timestep_device = device
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, timestep_device, timesteps, sigmas=sigmas
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latents
        transformer_dtype = self.transformer.dtype
        if use_input_image_size_as_output:
            height, width = processed_data["input_pixel_values"][0].shape[-2:]
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 8. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * (num_cfg + 1))
                latent_model_input = latent_model_input.to(transformer_dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    input_ids=processed_data["input_ids"],
                    input_img_latents=input_img_latents,
                    input_image_sizes=processed_data["input_image_sizes"],
                    attention_mask=processed_data["attention_mask"],
                    position_ids=processed_data["position_ids"],
                    return_dict=False,
                )[0]

                if num_cfg == 2:
                    cond, uncond, img_cond = torch.split(noise_pred, len(noise_pred) // 3, dim=0)
                    noise_pred = uncond + img_guidance_scale * (img_cond - uncond) + guidance_scale * (cond - img_cond)
                else:
                    cond, uncond = torch.split(noise_pred, len(noise_pred) // 2, dim=0)
                    noise_pred = uncond + guidance_scale * (cond - uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
