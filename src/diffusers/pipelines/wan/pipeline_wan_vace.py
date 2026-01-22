import html
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import regex as re
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import WanLoraLoaderMixin
from ...models import AutoencoderKLWan, WanVACETransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import WanPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

# Copied from diffusers.pipelines.stable_diffusion...
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

class WanVACEPipeline(DiffusionPipeline, WanLoraLoaderMixin):


    model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: WanVACETransformer3DModel = None,
        transformer_2: WanVACETransformer3DModel = None,
        boundary_ratio: Optional[float] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            transformer_2=transformer_2,
            scheduler=scheduler,
        )
        self.register_to_config(boundary_ratio=boundary_ratio)
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        
        """r"""
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
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

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        video=None,
        mask=None,
        reference_images=None,
        guidance_scale_2=None,
    ):
        if self.transformer is not None:
            base = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
        elif self.transformer_2 is not None:
            base = self.vae_scale_factor_spatial * self.transformer_2.config.patch_size[1]
        else:
            raise ValueError(
                "`transformer` or `transformer_2` component must be set in order to run inference with this pipeline"
            )

        if height % base != 0 or width % base != 0:
            raise ValueError(f"`height` and `width` have to be divisible by {base} but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if video is not None:
            if mask is not None:
                if len(video) != len(mask):
                    raise ValueError(
                        f"Length of `video` {len(video)} and `mask` {len(mask)} do not match. Please make sure that"
                        " they have the same length."
                    )
            if reference_images is not None:
                is_pil_image = isinstance(reference_images, PIL.Image.Image)
                is_list_of_pil_images = isinstance(reference_images, list) and all(
                    isinstance(ref_img, PIL.Image.Image) for ref_img in reference_images
                )
                is_list_of_list_of_pil_images = isinstance(reference_images, list) and all(
                    isinstance(ref_img, list) and all(isinstance(ref_img_, PIL.Image.Image) for ref_img_ in ref_img)
                    for ref_img in reference_images
                )
                if not (is_pil_image or is_list_of_pil_images or is_list_of_list_of_pil_images):
                    raise ValueError(
                        "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
                        "`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
                    )
                if is_list_of_list_of_pil_images and len(reference_images) != 1:
                    raise ValueError(
                        "The pipeline only supports generating one video at a time at the moment. When passing a list "
                        "of list of reference images, where the outer list corresponds to the batch size and the inner "
                        "list corresponds to list of conditioning images per video, please make sure to only pass "
                        "one inner list of reference images (i.e., `[[<image1>, <image2>, ...]]`"
                    )
        elif mask is not None:
            raise ValueError("`mask` can only be passed if `video` is passed as well.")

    def preprocess_conditions(
        self,
        video: Optional[List[PipelineImageInput]] = None,
        mask: Optional[List[PipelineImageInput]] = None,
        reference_images: Optional[Union[PIL.Image.Image, List[PIL.Image.Image], List[List[PIL.Image.Image]]]] = None,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        if video is not None:
            base = self.vae_scale_factor_spatial * (
                self.transformer.config.patch_size[1]
                if self.transformer is not None
                else self.transformer_2.config.patch_size[1]
            )
            video_height, video_width = self.video_processor.get_default_height_width(video[0])

            if video_height * video_width > height * width:
                scale = min(width / video_width, height / video_height)
                video_height, video_width = int(video_height * scale), int(video_width * scale)

            if video_height % base != 0 or video_width % base != 0:
                logger.warning(
                    f"Video height and width should be divisible by {base}, but got {video_height} and {video_width}. "
                )
                video_height = (video_height // base) * base
                video_width = (video_width // base) * base

            assert video_height * video_width <= height * width

            video = self.video_processor.preprocess_video(video, video_height, video_width)
            image_size = (video_height, video_width)  # Use the height/width of video (with possible rescaling)
        else:
            video = torch.zeros(batch_size, 3, num_frames, height, width, dtype=dtype, device=device)
            image_size = (height, width)  # Use the height/width provider by user

        if mask is not None:
            mask = self.video_processor.preprocess_video(mask, image_size[0], image_size[1])
            mask = torch.clamp((mask + 1) / 2, min=0, max=1)
        else:
            mask = torch.ones_like(video)

        video = video.to(dtype=dtype, device=device)
        mask = mask.to(dtype=dtype, device=device)

        # Make a list of list of images where the...
        # corresponds to list of conditioning images per video
        if reference_images is None or isinstance(reference_images, PIL.Image.Image):
            reference_images = [[reference_images] for _ in range(video.shape[0])]
        elif isinstance(reference_images, (list, tuple)) and isinstance(next(iter(reference_images)), PIL.Image.Image):
            reference_images = [reference_images]
        elif (
            isinstance(reference_images, (list, tuple))
            and isinstance(next(iter(reference_images)), list)
            and isinstance(next(iter(reference_images[0])), PIL.Image.Image)
        ):
            reference_images = reference_images
        else:
            raise ValueError(
                "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
                "`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
            )

        if video.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
            )

        ref_images_lengths = [len(reference_images_batch) for reference_images_batch in reference_images]
        if any(l != ref_images_lengths[0] for l in ref_images_lengths):
            raise ValueError(
                f"All batches of `reference_images` should have the same length, but got {ref_images_lengths}. Support for this "
                "may be added in the future."
            )

        reference_images_preprocessed = []
        for i, reference_images_batch in enumerate(reference_images):
            preprocessed_images = []
            for j, image in enumerate(reference_images_batch):
                if image is None:
                    continue
                image = self.video_processor.preprocess(image, None, None)
                img_height, img_width = image.shape[-2:]
                scale = min(image_size[0] / img_height, image_size[1] / img_width)
                new_height, new_width = int(img_height * scale), int(img_width * scale)
                resized_image = torch.nn.functional.interpolate(
                    image, size=(new_height, new_width), mode="bilinear", align_corners=False
                ).squeeze(0)  # [C, H, W]
                top = (image_size[0] - new_height) // 2
                left = (image_size[1] - new_width) // 2
                canvas = torch.ones(3, *image_size, device=device, dtype=dtype)
                canvas[:, top : top + new_height, left : left + new_width] = resized_image
                preprocessed_images.append(canvas)
            reference_images_preprocessed.append(preprocessed_images)

        return video, mask, reference_images_preprocessed

    def prepare_video_latents(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        reference_images: Optional[List[List[torch.Tensor]]] = None,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or self._execution_device

        if isinstance(generator, list):
            # TODO: support this
            raise ValueError("Passing a list of generators is not yet supported. This may be supported in the future.")

        if reference_images is None:
            # For each batch of video, we set no re
            # ference image (as one or more can be passed by user)
            reference_images = [[None] for _ in range(video.shape[0])]
        else:
            if video.shape[0] != len(reference_images):
                raise ValueError(
                    f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
                )

        if video.shape[0] != 1:
            # TODO: support this
            raise ValueError(
                "Generating with more than one video is not yet supported. This may be supported in the future."
            )

        vae_dtype = self.vae.dtype
        video = video.to(dtype=vae_dtype)

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )

        if mask is None:
            latents = retrieve_latents(self.vae.encode(video), generator, sample_mode="argmax").unbind(0)
            latents = ((latents.float() - latents_mean) * latents_std).to(vae_dtype)
        else:
            mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
            inactive = video * (1 - mask)
            reactive = video * mask
            inactive = retrieve_latents(self.vae.encode(inactive), generator, sample_mode="argmax")
            reactive = retrieve_latents(self.vae.encode(reactive), generator, sample_mode="argmax")
            inactive = ((inactive.float() - latents_mean) * latents_std).to(vae_dtype)
            reactive = ((reactive.float() - latents_mean) * latents_std).to(vae_dtype)
            latents = torch.cat([inactive, reactive], dim=1)

        latent_list = []
        for latent, reference_images_batch in zip(latents, reference_images):
            for reference_image in reference_images_batch:
                assert reference_image.ndim == 3
                reference_image = reference_image.to(dtype=vae_dtype)
                reference_image = reference_image[None, :, None, :, :]  # [1, C, 1, H, W]
                reference_latent = retrieve_latents(self.vae.encode(reference_image), generator, sample_mode="argmax")
                reference_latent = ((reference_latent.float() - latents_mean) * latents_std).to(vae_dtype)
                reference_latent = reference_latent.squeeze(0)  # [C, 1, H, W]
                reference_latent = torch.cat([reference_latent, torch.zeros_like(reference_latent)], dim=0)
                latent = torch.cat([reference_latent.squeeze(0), latent], dim=1)
            latent_list.append(latent)
        return torch.stack(latent_list)

    def prepare_masks(
        self,
        mask: torch.Tensor,
        reference_images: Optional[List[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if isinstance(generator, list):
            # TODO: support this
            raise ValueError("Passing a list of generators is not yet supported. This may be supported in the future.")

        if reference_images is None:
            # For each batch of video, we set no r...
            reference_images = [[None] for _ in range(mask.shape[0])]
        else:
            if mask.shape[0] != len(reference_images):
                raise ValueError(
                    f"Batch size of `mask` {mask.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
                )

        if mask.shape[0] != 1:
            # TODO: support this
            raise ValueError(
                "Generating with more than one video is not yet supported. This may be supported in the future."
            )

        transformer_patch_size = (
            self.transformer.config.patch_size[1]
            if self.transformer is not None
            else self.transformer_2.config.patch_size[1]
        )

        mask_list = []
        for mask_, reference_images_batch in zip(mask, reference_images):
            num_channels, num_frames, height, width = mask_.shape
            new_num_frames = (num_frames + self.vae_scale_factor_temporal - 1) // self.vae_scale_factor_temporal
            new_height = height // (self.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
            new_width = width // (self.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
            mask_ = mask_[0, :, :, :]
            mask_ = mask_.view(
                num_frames, new_height, self.vae_scale_factor_spatial, new_width, self.vae_scale_factor_spatial
            )
            mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(0, 1)  # [8x8, num_frames, new_height, new_width]
            mask_ = torch.nn.functional.interpolate(
                mask_.unsqueeze(0), size=(new_num_frames, new_height, new_width), mode="nearest-exact"
            ).squeeze(0)
            num_ref_images = len(reference_images_batch)
            if num_ref_images > 0:
                mask_padding = torch.zeros_like(mask_[:, :num_ref_images, :, :])
                mask_ = torch.cat([mask_padding, mask_], dim=1)
            mask_list.append(mask_)
        return torch.stack(mask_list)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
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
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        video: Optional[List[PipelineImageInput]] = None,
        mask: Optional[List[PipelineImageInput]] = None,
        reference_images: Optional[List[PipelineImageInput]] = None,
        conditioning_scale: Union[float, List[float], torch.Tensor] = 1.0,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):

        The call function to the pipeline for generation.
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            video (`List[PIL.Image.Image]`, *optional*):
                The input video or videos to be used as a starting point for the generation. The video should be a list
                of PIL images, a numpy array, or a torch tensor. Currently, the pipeline only supports generating one
                video at a time.
            mask (`List[PIL.Image.Image]`, *optional*):
                The input mask defines which video regions to condition on and which to generate. Black areas in the
                mask indicate conditioning regions, while white areas indicate regions for generation. The mask should
                be a list of PIL images, a numpy array, or a torch tensor. Currently supports generating a single video
                at a time.
            reference_images (`List[PIL.Image.Image]`, *optional*):
                A list of one or more reference images as extra conditioning for the generation. For example, if you
                are trying to inpaint a video to change the character, you can pass reference images of the new
                character here. Refer to the Diffusers [examples](https://github.com/huggingface/diffusers/pull/11582)
                and original [user
                guide](https://github.com/ali-vilab/VACE/blob/0897c6d055d7d9ea9e191dce763006664d9780f8/UserGuide.md)
                for a full list of supported tasks and use cases.
            conditioning_scale (`float`, `List[float]`, `torch.Tensor`, defaults to `1.0`):
                The conditioning scale to be applied when adding the control conditioning latent stream to the
                denoising latent stream in each control layer of the model. If a float is provided, it will be applied
                uniformly to all layers. If a list or tensor is provided, it should have the same length as the number
                of control layers in the model (`len(transformer.config.vace_layers)`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:
