import inspect
from typing import Any, List, Optional, Tuple, Union

import PIL
import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, ControlNetModel, ControlNetUnionModel, UNet2DConditionModel
from ...models.controlnets.multicontrolnet import MultiControlNetModel
from ...schedulers import EulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor, unwrap_module
from ..modular_pipeline import (
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import StableDiffusionXLModularPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# TODO(yiyi, aryan): We need another step before text encoder to set the `num_inference_steps` attribute for guider so that
# things like when to do guidance and how many con...
# always assuming you want to do guidance in the G...
# configuration of guider is.

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    
    """r"""
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        block_state.vae_scale_factor = components.vae_scale_factor

        block_state.height, block_state.width = block_state.latents.shape[-2:]
        block_state.height = block_state.height * block_state.vae_scale_factor
        block_state.width = block_state.width * block_state.vae_scale_factor

        block_state.original_size = block_state.original_size or (block_state.height, block_state.width)
        block_state.target_size = block_state.target_size or (block_state.height, block_state.width)

        block_state.text_encoder_projection_dim = int(block_state.pooled_prompt_embeds.shape[-1])

        if block_state.negative_original_size is None:
            block_state.negative_original_size = block_state.original_size
        if block_state.negative_target_size is None:
            block_state.negative_target_size = block_state.target_size

        block_state.add_time_ids, block_state.negative_add_time_ids = self._get_add_time_ids(
            components,
            block_state.original_size,
            block_state.crops_coords_top_left,
            block_state.target_size,
            block_state.aesthetic_score,
            block_state.negative_aesthetic_score,
            block_state.negative_original_size,
            block_state.negative_crops_coords_top_left,
            block_state.negative_target_size,
            dtype=block_state.pooled_prompt_embeds.dtype,
            text_encoder_projection_dim=block_state.text_encoder_projection_dim,
        )
        block_state.add_time_ids = block_state.add_time_ids.repeat(
            block_state.batch_size * block_state.num_images_per_prompt, 1
        ).to(device=block_state.device)
        block_state.negative_add_time_ids = block_state.negative_add_time_ids.repeat(
            block_state.batch_size * block_state.num_images_per_prompt, 1
        ).to(device=block_state.device)

        # Optionally get Guidance Scale Embedding for LCM
        block_state.timestep_cond = None
        if (
            hasattr(components, "unet")
            and components.unet is not None
            and components.unet.config.time_cond_proj_dim is not None
        ):
            # TODO(yiyi, aryan): Ideally, this should be `embedded_guidance_scale` instead of pulling from guider. Guider scales should be different from this!
            block_state.guidance_scale_tensor = torch.tensor(components.guider.guidance_scale - 1).repeat(
                block_state.batch_size * block_state.num_images_per_prompt
            )
            block_state.timestep_cond = self.get_guidance_scale_embedding(
                block_state.guidance_scale_tensor, embedding_dim=components.unet.config.time_cond_proj_dim
            ).to(device=block_state.device, dtype=block_state.latents.dtype)

        self.set_block_state(state, block_state)
        return components, state

class StableDiffusionXLPrepareAdditionalConditioningStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return "Step that prepares the additional conditioning for the text-to-image generation process"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("original_size"),
            InputParam("target_size"),
            InputParam("negative_original_size"),
            InputParam("negative_target_size"),
            InputParam("crops_coords_top_left", default=(0, 0)),
            InputParam("negative_crops_coords_top_left", default=(0, 0)),
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "pooled_prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="The pooled prompt embeddings to use for the denoising process (used to determine shapes and dtypes for other additional conditioning inputs). Can be generated in text_encoder step.",
            ),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "add_time_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The time ids to condition the denoising process",
            ),
            OutputParam(
                "negative_add_time_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The negative time ids to condition the denoising process",
            ),
            OutputParam("timestep_cond", type_hint=torch.Tensor, description="The timestep cond to use for LCM"),
        ]

    @staticmethod
    # Copied from diffusers.pipelines.stable_diffu...
    def _get_add_time_ids(
        components, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            components.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = components.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.latent_consi...
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        class StableDiffusionXLPrepareAdditionalConditioningStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return "Step that prepares the additional conditioning for the text-to-image generation process"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("original_size"),
            InputParam("target_size"),
            InputParam("negative_original_size"),
            InputParam("negative_target_size"),
            InputParam("crops_coords_top_left", default=(0, 0)),
            InputParam("negative_crops_coords_top_left", default=(0, 0)),
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "pooled_prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="The pooled prompt embeddings to use for the denoising process (used to determine shapes and dtypes for other additional conditioning inputs). Can be generated in text_encoder step.",
            ),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "add_time_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The time ids to condition the denoising process",
            ),
            OutputParam(
                "negative_add_time_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The negative time ids to condition the denoising process",
            ),
            OutputParam("timestep_cond", type_hint=torch.Tensor, description="The timestep cond to use for LCM"),
        ]

    @staticmethod
    # Copied from diffusers.pipelines.stable_diffu...
    def _get_add_time_ids(
        components, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            components.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = components.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.latent_consi...
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:

        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        block_state.height, block_state.width = block_state.latents.shape[-2:]
        block_state.height = block_state.height * components.vae_scale_factor
        block_state.width = block_state.width * components.vae_scale_factor

        block_state.original_size = block_state.original_size or (block_state.height, block_state.width)
        block_state.target_size = block_state.target_size or (block_state.height, block_state.width)

        block_state.text_encoder_projection_dim = int(block_state.pooled_prompt_embeds.shape[-1])

        block_state.add_time_ids = self._get_add_time_ids(
            components,
            block_state.original_size,
            block_state.crops_coords_top_left,
            block_state.target_size,
            block_state.pooled_prompt_embeds.dtype,
            text_encoder_projection_dim=block_state.text_encoder_projection_dim,
        )
        if block_state.negative_original_size is not None and block_state.negative_target_size is not None:
            block_state.negative_add_time_ids = self._get_add_time_ids(
                components,
                block_state.negative_original_size,
                block_state.negative_crops_coords_top_left,
                block_state.negative_target_size,
                block_state.pooled_prompt_embeds.dtype,
                text_encoder_projection_dim=block_state.text_encoder_projection_dim,
            )
        else:
            block_state.negative_add_time_ids = block_state.add_time_ids

        block_state.add_time_ids = block_state.add_time_ids.repeat(
            block_state.batch_size * block_state.num_images_per_prompt, 1
        ).to(device=block_state.device)
        block_state.negative_add_time_ids = block_state.negative_add_time_ids.repeat(
            block_state.batch_size * block_state.num_images_per_prompt, 1
        ).to(device=block_state.device)

        # Optionally get Guidance Scale Embedding for LCM
        block_state.timestep_cond = None
        if (
            hasattr(components, "unet")
            and components.unet is not None
            and components.unet.config.time_cond_proj_dim is not None
        ):
            # TODO(yiyi, aryan): Ideally, this should be `embedded_guidance_scale` instead of pulling from guider. Guider scales should be different from this!
            block_state.guidance_scale_tensor = torch.tensor(components.guider.guidance_scale - 1).repeat(
                block_state.batch_size * block_state.num_images_per_prompt
            )
            block_state.timestep_cond = self.get_guidance_scale_embedding(
                block_state.guidance_scale_tensor, embedding_dim=components.unet.config.time_cond_proj_dim
            ).to(device=block_state.device, dtype=block_state.latents.dtype)

        self.set_block_state(state, block_state)
        return components, state

class StableDiffusionXLControlNetInputStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("controlnet", ControlNetModel),
            ComponentSpec(
                "control_image_processor",
                VaeImageProcessor,
                config=FrozenDict({"do_convert_rgb": True, "do_normalize": False}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "step that prepare inputs for controlnet"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("control_image", required=True),
            InputParam("control_guidance_start", default=0.0),
            InputParam("control_guidance_end", default=1.0),
            InputParam("controlnet_conditioning_scale", default=1.0),
            InputParam("guess_mode", default=False),
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
            InputParam(
                "crops_coords",
                type_hint=Optional[Tuple[int]],
                description="The crop coordinates to use for preprocess/postprocess the image and mask, for inpainting task only. Can be generated in vae_encode step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("controlnet_cond", type_hint=torch.Tensor, description="The processed control image"),
            OutputParam(
                "control_guidance_start", type_hint=List[float], description="The controlnet guidance start values"
            ),
            OutputParam(
                "control_guidance_end", type_hint=List[float], description="The controlnet guidance end values"
            ),
            OutputParam(
                "conditioning_scale", type_hint=List[float], description="The controlnet conditioning scale values"
            ),
            OutputParam("guess_mode", type_hint=bool, description="Whether guess mode is used"),
            OutputParam("controlnet_keep", type_hint=List[float], description="The controlnet keep values"),
        ]

    # Modified from diffusers.pipelines.controlnet...
    # 1. return image without apply any guidance
    # 2. add crops_coords and resize_mode to preprocess()
    @staticmethod
    def prepare_control_image(
        components,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        crops_coords=None,
    ):
        if crops_coords is not None:
            image = components.control_image_processor.preprocess(
                image, height=height, width=width, crops_coords=crops_coords, resize_mode="fill"
            ).to(dtype=torch.float32)
        else:
            image = components.control_image_processor.preprocess(image, height=height, width=width).to(
                dtype=torch.float32
            )

        image_batch_size = image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)
        return image

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # (1) prepare controlnet inputs
        block_state.device = components._execution_device
        block_state.height, block_state.width = block_state.latents.shape[-2:]
        block_state.height = block_state.height * components.vae_scale_factor
        block_state.width = block_state.width * components.vae_scale_factor

        controlnet = unwrap_module(components.controlnet)

        # (1.1)
        # control_guidance_start/control_guidance_end (align format)
        if not isinstance(block_state.control_guidance_start, list) and isinstance(
            block_state.control_guidance_end, list
        ):
            block_state.control_guidance_start = len(block_state.control_guidance_end) * [
                block_state.control_guidance_start
            ]
        elif not isinstance(block_state.control_guidance_end, list) and isinstance(
            block_state.control_guidance_start, list
        ):
            block_state.control_guidance_end = len(block_state.control_guidance_start) * [
                block_state.control_guidance_end
            ]
        elif not isinstance(block_state.control_guidance_start, list) and not isinstance(
            block_state.control_guidance_end, list
        ):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            block_state.control_guidance_start, block_state.control_guidance_end = (
                mult * [block_state.control_guidance_start],
                mult * [block_state.control_guidance_end],
            )

        # (1.2)
        # controlnet_conditioning_scale (align format)
        if isinstance(controlnet, MultiControlNetModel) and isinstance(
            block_state.controlnet_conditioning_scale, float
        ):
            block_state.controlnet_conditioning_scale = [block_state.controlnet_conditioning_scale] * len(
                controlnet.nets
            )

        # (1.3)
        # global_pool_conditions
        block_state.global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        # (1.4)
        # guess_mode
        block_state.guess_mode = block_state.guess_mode or block_state.global_pool_conditions

        # (1.5)
        # control_image
        if isinstance(controlnet, ControlNetModel):
            block_state.control_image = self.prepare_control_image(
                components,
                image=block_state.control_image,
                width=block_state.width,
                height=block_state.height,
                batch_size=block_state.batch_size * block_state.num_images_per_prompt,
                num_images_per_prompt=block_state.num_images_per_prompt,
                device=block_state.device,
                dtype=controlnet.dtype,
                crops_coords=block_state.crops_coords,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in block_state.control_image:
                control_image = self.prepare_control_image(
                    components,
                    image=control_image_,
                    width=block_state.width,
                    height=block_state.height,
                    batch_size=block_state.batch_size * block_state.num_images_per_prompt,
                    num_images_per_prompt=block_state.num_images_per_prompt,
                    device=block_state.device,
                    dtype=controlnet.dtype,
                    crops_coords=block_state.crops_coords,
                )

                control_images.append(control_image)

            block_state.control_image = control_images
        else:
            assert False

        # (1.6)
        # controlnet_keep
        block_state.controlnet_keep = []
        for i in range(len(block_state.timesteps)):
            keeps = [
                1.0 - float(i / len(block_state.timesteps) < s or (i + 1) / len(block_state.timesteps) > e)
                for s, e in zip(block_state.control_guidance_start, block_state.control_guidance_end)
            ]
            block_state.controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        block_state.controlnet_cond = block_state.control_image
        block_state.conditioning_scale = block_state.controlnet_conditioning_scale

        self.set_block_state(state, block_state)

        return components, state

class StableDiffusionXLControlNetUnionInputStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("controlnet", ControlNetUnionModel),
            ComponentSpec(
                "control_image_processor",
                VaeImageProcessor,
                config=FrozenDict({"do_convert_rgb": True, "do_normalize": False}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "step that prepares inputs for the ControlNetUnion model"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("control_image", required=True),
            InputParam("control_mode", required=True),
            InputParam("control_guidance_start", default=0.0),
            InputParam("control_guidance_end", default=1.0),
            InputParam("controlnet_conditioning_scale", default=1.0),
            InputParam("guess_mode", default=False),
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Used to determine the shape of the control images. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
            InputParam(
                "dtype",
                required=True,
                type_hint=torch.dtype,
                description="The dtype of model tensor inputs. Can be generated in input step.",
            ),
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Needed to determine `controlnet_keep`. Can be generated in set_timesteps step.",
            ),
            InputParam(
                "crops_coords",
                type_hint=Optional[Tuple[int]],
                description="The crop coordinates to use for preprocess/postprocess the image and mask, for inpainting task only. Can be generated in vae_encode step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("controlnet_cond", type_hint=List[torch.Tensor], description="The processed control images"),
            OutputParam(
                "control_type_idx",
                type_hint=List[int],
                description="The control mode indices",
                kwargs_type="controlnet_kwargs",
            ),
            OutputParam(
                "control_type",
                type_hint=torch.Tensor,
                description="The control type tensor that specifies which control type is active",
                kwargs_type="controlnet_kwargs",
            ),
            OutputParam("control_guidance_start", type_hint=float, description="The controlnet guidance start value"),
            OutputParam("control_guidance_end", type_hint=float, description="The controlnet guidance end value"),
            OutputParam(
                "conditioning_scale", type_hint=List[float], description="The controlnet conditioning scale values"
            ),
            OutputParam("guess_mode", type_hint=bool, description="Whether guess mode is used"),
            OutputParam("controlnet_keep", type_hint=List[float], description="The controlnet keep values"),
        ]

    # Modified from diffusers.pipelines.controlnet...
    # 1. return image without apply any guidance
    # 2. add crops_coords and resize_mode to preprocess()
    @staticmethod
    def prepare_control_image(
        components,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        crops_coords=None,
    ):
        if crops_coords is not None:
            image = components.control_image_processor.preprocess(
                image, height=height, width=width, crops_coords=crops_coords, resize_mode="fill"
            ).to(dtype=torch.float32)
        else:
            image = components.control_image_processor.preprocess(image, height=height, width=width).to(
                dtype=torch.float32
            )

        image_batch_size = image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)
        return image

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        controlnet = unwrap_module(components.controlnet)

        device = components._execution_device
        dtype = block_state.dtype or components.controlnet.dtype

        block_state.height, block_state.width = block_state.latents.shape[-2:]
        block_state.height = block_state.height * components.vae_scale_factor
        block_state.width = block_state.width * components.vae_scale_factor

        # control_guidance_start/control_guidance_end (align format)
        if not isinstance(block_state.control_guidance_start, list) and isinstance(
            block_state.control_guidance_end, list
        ):
            block_state.control_guidance_start = len(block_state.control_guidance_end) * [
                block_state.control_guidance_start
            ]
        elif not isinstance(block_state.control_guidance_end, list) and isinstance(
            block_state.control_guidance_start, list
        ):
            block_state.control_guidance_end = len(block_state.control_guidance_start) * [
                block_state.control_guidance_end
            ]

        # guess_mode
        block_state.global_pool_conditions = controlnet.config.global_pool_conditions
        block_state.guess_mode = block_state.guess_mode or block_state.global_pool_conditions

        # control_image
        if not isinstance(block_state.control_image, list):
            block_state.control_image = [block_state.control_image]
        # control_mode
        if not isinstance(block_state.control_mode, list):
            block_state.control_mode = [block_state.control_mode]

        if len(block_state.control_image) != len(block_state.control_mode):
            raise ValueError("Expected len(control_image) == len(control_type)")

        # control_type
        block_state.num_control_type = controlnet.config.num_control_type
        block_state.control_type = [0 for _ in range(block_state.num_control_type)]
        for control_idx in block_state.control_mode:
            block_state.control_type[control_idx] = 1
        block_state.control_type = torch.Tensor(block_state.control_type)

        block_state.control_type = block_state.control_type.reshape(1, -1).to(device, dtype=block_state.dtype)
        repeat_by = block_state.batch_size * block_state.num_images_per_prompt // block_state.control_type.shape[0]
        block_state.control_type = block_state.control_type.repeat_interleave(repeat_by, dim=0)

        # prepare control_image
        for idx, _ in enumerate(block_state.control_image):
            block_state.control_image[idx] = self.prepare_control_image(
                components,
                image=block_state.control_image[idx],
                width=block_state.width,
                height=block_state.height,
                batch_size=block_state.batch_size * block_state.num_images_per_prompt,
                num_images_per_prompt=block_state.num_images_per_prompt,
                device=device,
                dtype=dtype,
                crops_coords=block_state.crops_coords,
            )
            block_state.height, block_state.width = block_state.control_image[idx].shape[-2:]

        # controlnet_keep
        block_state.controlnet_keep = []
        for i in range(len(block_state.timesteps)):
            block_state.controlnet_keep.append(
                1.0
                - float(
                    i / len(block_state.timesteps) < block_state.control_guidance_start
                    or (i + 1) / len(block_state.timesteps) > block_state.control_guidance_end
                )
            )
        block_state.control_type_idx = block_state.control_mode
        block_state.controlnet_cond = block_state.control_image
        block_state.conditioning_scale = block_state.controlnet_conditioning_scale

        self.set_block_state(state, block_state)

        return components, state
