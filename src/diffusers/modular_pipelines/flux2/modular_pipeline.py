from ...loaders import Flux2LoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class Flux2ModularPipeline(ModularPipeline, Flux2LoraLoaderMixin):


    default_blocks_name = "Flux2AutoBlocks"

    @property
    def default_height(self):
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_width(self):
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_sample_size(self):
        return 128

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 8
        if getattr(self, "vae", None) is not None:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return vae_scale_factor

    @property
    def num_channels_latents(self):
        num_channels_latents = 32
        if getattr(self, "transformer", None):
            num_channels_latents = self.transformer.config.in_channels // 4
        return num_channels_latents
