import numpy as np

from ...configuration_utils import register_to_config
from ...video_processor import VideoProcessor

# copied from https://github.com/Tencent-Hunyuan/H...
def generate_crop_size_list(base_size=256, patch_size=16, max_ratio=4.0):
    num_patches = round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list

# copied from https://github.com/Tencent-Hunyuan/H...
def get_closest_ratio(height: float, width: float, ratios: list, buckets: list):

    aspect_ratio = float(height) / float(width)
    diff_ratios = ratios - aspect_ratio

    if aspect_ratio >= 1:
        indices = [(index, x) for index, x in enumerate(diff_ratios) if x <= 0]
    else:
        indices = [(index, x) for index, x in enumerate(diff_ratios) if x >= 0]

    closest_ratio_id = min(indices, key=lambda pair: abs(pair[1]))[0]
    closest_size = buckets[closest_ratio_id]
    closest_ratio = ratios[closest_ratio_id]

    return closest_size, closest_ratio

class HunyuanVideo15ImageProcessor(VideoProcessor):


    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 16,
        vae_latent_channels: int = 32,
        do_convert_rgb: bool = True,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            do_convert_rgb=do_convert_rgb,
        )

    def calculate_default_height_width(self, height: int, width: int, target_size: int):
        crop_size_list = generate_crop_size_list(base_size=target_size, patch_size=self.config.vae_scale_factor)
        aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
        height, width = get_closest_ratio(height, width, aspect_ratios, crop_size_list)[0]

        return height, width
