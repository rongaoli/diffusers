from typing import List, Optional, Tuple, Union

import torch

from ...models import UNet1DModel
from ...schedulers import SchedulerMixin
from ...utils import is_torch_xla_available, logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DeprecatedPipelineMixin, DiffusionPipeline

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class DanceDiffusionPipeline(DeprecatedPipelineMixin, DiffusionPipeline):


    _last_supported_version = "0.33.1"
    model_cpu_offload_seq = "unet"

    def __init__(self, unet: UNet1DModel, scheduler: SchedulerMixin):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 100,
        generator: Optional[torch.Generator] = None,
        audio_length_in_s: Optional[float] = None,
        return_dict: bool = True,
    ) -> Union[AudioPipelineOutput, Tuple]:
        
        """r"""

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size / self.unet.config.sample_rate

        sample_size = audio_length_in_s * self.unet.config.sample_rate

        down_scale_factor = 2 ** len(self.unet.up_blocks)
        if sample_size < 3 * down_scale_factor:
            raise ValueError(
                f"{audio_length_in_s} is too small. Make sure it's bigger or equal to"
                f" {3 * down_scale_factor / self.unet.config.sample_rate}."
            )

        original_sample_size = int(sample_size)
        if sample_size % down_scale_factor != 0:
            sample_size = (
                (audio_length_in_s * self.unet.config.sample_rate) // down_scale_factor + 1
            ) * down_scale_factor
            logger.info(
                f"{audio_length_in_s} is increased to {sample_size / self.unet.config.sample_rate} so that it can be handled"
                f" by the model. It will be cut to {original_sample_size / self.unet.config.sample_rate} after the denoising"
                " process."
            )
        sample_size = int(sample_size)

        dtype = next(self.unet.parameters()).dtype
        shape = (batch_size, self.unet.config.in_channels, sample_size)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        audio = randn_tensor(shape, generator=generator, device=self._execution_device, dtype=dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps, device=audio.device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(dtype)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(audio, t).sample

            # 2. compute previous audio sample: x_t -> t_t-1
            audio = self.scheduler.step(model_output, t, audio).prev_sample

            if XLA_AVAILABLE:
                xm.mark_step()

        audio = audio.clamp(-1, 1).float().cpu().numpy()

        audio = audio[:, :, :original_sample_size]

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)
