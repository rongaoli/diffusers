from typing import Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from ...configuration_utils import register_to_config
from ...image_processor import VaeImageProcessor
from ...utils import PIL_INTERPOLATION

class WanAnimateImageProcessor(VaeImageProcessor):


    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        vae_latent_channels: int = 16,
        spatial_patch_size: Tuple[int, int] = (2, 2),
        resample: str = "lanczos",
        reducing_gap: int = None,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
        fill_color: Optional[Union[str, float, Tuple[float, ...]]] = 0,
    ):
        super().__init__()
        if do_convert_rgb and do_convert_grayscale:
            raise ValueError(
                "`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`,"
                " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`.",
                " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`",
            )

    def _resize_and_fill(
        self,
        image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:
        
        """r"""

        ratio = width / height
        src_ratio = image.width / image.height
        fill_with_image_data = self.config.fill_color is None
        fill_color = self.config.fill_color or 0

        src_w = width if ratio < src_ratio else image.width * height // image.height
        src_h = height if ratio >= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION[self.config.resample])
        res = PIL.Image.new("RGB", (width, height), color=fill_color)
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if fill_with_image_data:
            if ratio < src_ratio:
                fill_height = height // 2 - src_h // 2
                if fill_height > 0:
                    res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                    res.paste(
                        resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                        box=(0, fill_height + src_h),
                    )
            elif ratio > src_ratio:
                fill_width = width // 2 - src_w // 2
                if fill_width > 0:
                    res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                    res.paste(
                        resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                        box=(fill_width + src_w, 0),
                    )

        return res

    def get_default_height_width(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[int, int]:
        
        """r"""

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]
            else:
                height = image.shape[1]

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]
            else:
                width = image.shape[2]

        max_area = width * height
        aspect_ratio = height / width
        mod_value_h = self.config.vae_scale_factor * self.config.spatial_patch_size[0]
        mod_value_w = self.config.vae_scale_factor * self.config.spatial_patch_size[1]

        # Try to preserve the aspect ratio
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value_h * mod_value_h
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value_w * mod_value_w

        return height, width
