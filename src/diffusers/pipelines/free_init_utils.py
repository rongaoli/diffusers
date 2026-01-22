import math
from typing import Tuple, Union

import torch
import torch.fft as fft

from ..utils.torch_utils import randn_tensor

class FreeInitMixin:


    def enable_free_init(
        self,
        num_iters: int = 3,
        use_fast_sampling: bool = False,
        method: str = "butterworth",
        order: int = 4,
        spatial_stop_frequency: float = 0.25,
        temporal_stop_frequency: float = 0.25,
    ):

        self._free_init_num_iters = num_iters
        self._free_init_use_fast_sampling = use_fast_sampling
        self._free_init_method = method
        self._free_init_order = order
        self._free_init_spatial_stop_frequency = spatial_stop_frequency
        self._free_init_temporal_stop_frequency = temporal_stop_frequency

    def disable_free_init(self):

        self._free_init_num_iters = None

    @property
    def free_init_enabled(self):
        return hasattr(self, "_free_init_num_iters") and self._free_init_num_iters is not None

    def _get_free_init_freq_filter(
        self,
        shape: Tuple[int, ...],
        device: Union[str, torch.dtype],
        filter_type: str,
        order: float,
        spatial_stop_frequency: float,
        temporal_stop_frequency: float,
    ) -> torch.Tensor:
        
        """r"""

        time, height, width = shape[-3], shape[-2], shape[-1]
        mask = torch.zeros(shape)

        if spatial_stop_frequency == 0 or temporal_stop_frequency == 0:
            return mask

        if filter_type == "butterworth":

            def retrieve_mask(x):
                return 1 / (1 + (x / spatial_stop_frequency**2) ** order)
        elif filter_type == "gaussian":

            def retrieve_mask(x):
                return math.exp(-1 / (2 * spatial_stop_frequency**2) * x)
        elif filter_type == "ideal":

            def retrieve_mask(x):
                return 1 if x <= spatial_stop_frequency * 2 else 0
        else:
            raise NotImplementedError("`filter_type` must be one of gaussian, butterworth or ideal")

        for t in range(time):
            for h in range(height):
                for w in range(width):
                    d_square = (
                        ((spatial_stop_frequency / temporal_stop_frequency) * (2 * t / time - 1)) ** 2
                        + (2 * h / height - 1) ** 2
                        + (2 * w / width - 1) ** 2
                    )
                    mask[..., t, h, w] = retrieve_mask(d_square)

        return mask.to(device)

    def _apply_freq_filter(self, x: torch.Tensor, noise: torch.Tensor, low_pass_filter: torch.Tensor) -> torch.Tensor:
        
        """r"""
        # FFT
        x_freq = fft.fftn(x, dim=(-3, -2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
        noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
        noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

        # frequency mix
        high_pass_filter = 1 - low_pass_filter
        x_freq_low = x_freq * low_pass_filter
        noise_freq_high = noise_freq * high_pass_filter
        x_freq_mixed = x_freq_low + noise_freq_high  # mix in freq domain

        # IFFT
        x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
        x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

        return x_mixed

    def _apply_free_init(
        self,
        latents: torch.Tensor,
        free_init_iteration: int,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ):
        if free_init_iteration == 0:
            self._free_init_initial_noise = latents.detach().clone()
        else:
            latent_shape = latents.shape

            free_init_filter_shape = (1, *latent_shape[1:])
            free_init_freq_filter = self._get_free_init_freq_filter(
                shape=free_init_filter_shape,
                device=device,
                filter_type=self._free_init_method,
                order=self._free_init_order,
                spatial_stop_frequency=self._free_init_spatial_stop_frequency,
                temporal_stop_frequency=self._free_init_temporal_stop_frequency,
            )

            current_diffuse_timestep = self.scheduler.config.num_train_timesteps - 1
            diffuse_timesteps = torch.full((latent_shape[0],), current_diffuse_timestep).long()

            z_t = self.scheduler.add_noise(
                original_samples=latents, noise=self._free_init_initial_noise, timesteps=diffuse_timesteps.to(device)
            ).to(dtype=torch.float32)

            z_rand = randn_tensor(
                shape=latent_shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            latents = self._apply_freq_filter(z_t, z_rand, low_pass_filter=free_init_freq_filter)
            latents = latents.to(dtype)

        # Coarse-to-Fine Sampling for faster inference (can lead to lower quality)
        if self._free_init_use_fast_sampling:
            num_inference_steps = max(
                1, int(num_inference_steps / self._free_init_num_iters * (free_init_iteration + 1))
            )

        if num_inference_steps > 0:
            self.scheduler.set_timesteps(num_inference_steps, device=device)

        return latents, self.scheduler.timesteps
