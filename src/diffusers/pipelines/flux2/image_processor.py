import math
from typing import List

import PIL.Image

from ...configuration_utils import register_to_config
from ...image_processor import VaeImageProcessor

class Flux2ImageProcessor(VaeImageProcessor):


    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 16,
        vae_latent_channels: int = 32,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            do_normalize=do_normalize,
            do_convert_rgb=do_convert_rgb,
        )

    @staticmethod
    def check_image_input(
        image: PIL.Image.Image, max_aspect_ratio: int = 8, min_side_length: int = 64, max_area: int = 1024 * 1024
    ) -> PIL.Image.Image:

        if not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Image must be a PIL.Image.Image, got {type(image)}")

        width, height = image.size

        # Check minimum dimensions
        if width < min_side_length or height < min_side_length:
            raise ValueError(
                f"Image too small: {width}×{height}. Both dimensions must be at least {min_side_length}px"
            )

        # Check aspect ratio
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            raise ValueError(
                f"Aspect ratio too extreme: {width}×{height} (ratio: {aspect_ratio:.1f}:1). "
                f"Maximum allowed ratio is {max_aspect_ratio}:1"
            )

        return image

    @staticmethod
    def _resize_to_target_area(image: PIL.Image.Image, target_area: int = 1024 * 1024) -> PIL.Image.Image:
        image_width, image_height = image.size

        scale = math.sqrt(target_area / (image_width * image_height))
        width = int(image_width * scale)
        height = int(image_height * scale)

        return image.resize((width, height), PIL.Image.Resampling.LANCZOS)

    @staticmethod
    def _resize_if_exceeds_area(image, target_area=1024 * 1024) -> PIL.Image.Image:
        image_width, image_height = image.size
        pixel_count = image_width * image_height
        if pixel_count <= target_area:
            return image
        return Flux2ImageProcessor._resize_to_target_area(image, target_area)

    def _resize_and_crop(
        self,
        image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:
        
        """r"""
        image_width, image_height = image.size

        left = (image_width - width) // 2
        top = (image_height - height) // 2
        right = left + width
        bottom = top + height

        return image.crop((left, top, right, bottom))

    # Taken from
    # https://github.com/black-forest-labs/flux2/b...
    @staticmethod
    def concatenate_images(images: List[PIL.Image.Image]) -> PIL.Image.Image:


        # If only one image, return a copy of it
        if len(images) == 1:
            return images[0].copy()

        # Convert all images to RGB if not already
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

        # Calculate dimensions for horizontal concatenation
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # Create new image with white background
        background_color = (255, 255, 255)
        new_img = PIL.Image.new("RGB", (total_width, max_height), background_color)

        # Paste images with center alignment
        x_offset = 0
        for img in images:
            y_offset = (max_height - img.height) // 2
            new_img.paste(img, (x_offset, y_offset))
            x_offset += img.width

        return new_img
