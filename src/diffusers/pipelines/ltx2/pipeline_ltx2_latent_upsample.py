from typing import List, Optional, Union

import torch

from ...image_processor import PipelineImageInput
from ...models import AutoencoderKLLTX2Video
from ...utils import get_logger, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..ltx.pipeline_output import LTXPipelineOutput
from ..pipeline_utils import DiffusionPipeline
from .latent_upsampler import LTX2LatentUpsamplerModel

logger = get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """


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

class LTX2LatentUpsamplePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "vae->latent_upsampler"

    def __init__(
        self,
        vae: AutoencoderKLLTX2Video,
        latent_upsampler: LTX2LatentUpsamplerModel,
    ) -> None:
        super().__init__()

        self.register_modules(vae=vae, latent_upsampler=latent_upsampler)

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

    def prepare_latents(
        self,
        video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        spatial_patch_size: int = 1,
        temporal_patch_size: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 3:
                # Convert token seq [B, S, D] to latent video [B, C, F, H, W]
                latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
                latent_height = height // self.vae_spatial_compression_ratio
                latent_width = width // self.vae_spatial_compression_ratio
                latents = self._unpack_latents(
                    latents, latent_num_frames, latent_height, latent_width, spatial_patch_size, temporal_patch_size
                )
            return latents.to(device=device, dtype=dtype)

        video = video.to(device=device, dtype=self.vae.dtype)
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            init_latents = [
                retrieve_latents(self.vae.encode(video[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]
        else:
            init_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in video]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        # NOTE: latent upsampler operates on the unnormalized latents, so don't normalize here
        # init_latents = self._normalize_latents(i...
        return init_latents

    def adain_filter_latent(self, latents: torch.Tensor, reference_latents: torch.Tensor, factor: float = 1.0):
        class LTX2LatentUpsamplePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "vae->latent_upsampler"

    def __init__(
        self,
        vae: AutoencoderKLLTX2Video,
        latent_upsampler: LTX2LatentUpsamplerModel,
    ) -> None:
        super().__init__()

        self.register_modules(vae=vae, latent_upsampler=latent_upsampler)

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

    def prepare_latents(
        self,
        video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        spatial_patch_size: int = 1,
        temporal_patch_size: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 3:
                # Convert token seq [B, S, D] to latent video [B, C, F, H, W]
                latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
                latent_height = height // self.vae_spatial_compression_ratio
                latent_width = width // self.vae_spatial_compression_ratio
                latents = self._unpack_latents(
                    latents, latent_num_frames, latent_height, latent_width, spatial_patch_size, temporal_patch_size
                )
            return latents.to(device=device, dtype=dtype)

        video = video.to(device=device, dtype=self.vae.dtype)
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            init_latents = [
                retrieve_latents(self.vae.encode(video[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]
        else:
            init_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in video]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        # NOTE: latent upsampler operates on the unnormalized latents, so don't normalize here
        # init_latents = self._normalize_latents(i...
        return init_latents

    def adain_filter_latent(self, latents: torch.Tensor, reference_latents: torch.Tensor, factor: float = 1.0):

        result = latents.clone()

        for i in range(latents.size(0)):
            for c in range(latents.size(1)):
                r_sd, r_mean = torch.std_mean(reference_latents[i, c], dim=None)  # index by original dim order
                i_sd, i_mean = torch.std_mean(result[i, c], dim=None)

                result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean

        result = torch.lerp(latents, result, factor)
        return result

    def tone_map_latents(self, latents: torch.Tensor, compression: float) -> torch.Tensor:
        Applies a non-linear tone-mapping function to latent values to reduce their dynamic range in a perceptually
        smooth way using a sigmoid-based compression.

        This is useful for regularizing high-variance latents or for conditioning outputs during generation, especially
        when controlling dynamic behavior with a `compression` factor.
                - 0.0: No tone-mapping (identity transform)
                - 1.0: Full compression effect
