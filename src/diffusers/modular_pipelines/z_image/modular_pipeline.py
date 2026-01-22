from ...loaders import ZImageLoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class ZImageModularPipeline(
    ModularPipeline,
    ZImageLoraLoaderMixin,
):


    default_blocks_name = "ZImageAutoBlocks"

    @property
    def default_height(self):
        return 1024

    @property
    def default_width(self):
        return 1024

    @property
    def vae_scale_factor_spatial(self):
        vae_scale_factor_spatial = 16
        if hasattr(self, "image_processor") and self.image_processor is not None:
            vae_scale_factor_spatial = self.image_processor.config.vae_scale_factor
        return vae_scale_factor_spatial

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 8
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return vae_scale_factor

    @property
    def num_channels_latents(self):
        num_channels_latents = 16
        if hasattr(self, "transformer") and self.transformer is not None:
            num_channels_latents = self.transformer.config.in_channels
        return num_channels_latents

    @property
    def requires_unconditional_embeds(self):
        requires_unconditional_embeds = False

        if hasattr(self, "guider") and self.guider is not None:
            requires_unconditional_embeds = self.guider._enabled and self.guider.num_conditions > 1

        return requires_unconditional_embeds
