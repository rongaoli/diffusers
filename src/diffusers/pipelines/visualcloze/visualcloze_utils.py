from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from ...image_processor import VaeImageProcessor

class VisualClozeProcessor(VaeImageProcessor):


    def __init__(self, *args, resolution: int = 384, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = resolution

    def preprocess_image(
        self, input_images: List[List[Optional[Image.Image]]], vae_scale_factor: int
    ) -> Tuple[List[List[torch.Tensor]], List[List[List[int]]], List[int]]:
        Preprocesses input images for the VisualCloze pipeline.

        This function handles the preprocessing of input images by:
        1. Resizing and cropping images to maintain consistent dimensions
        2. Converting images to the Tensor format for the VAE
        3. Normalizing pixel values
        4. Tracking image sizes and positions of target images
                - Outer list represents different samples, including in-context examples and the query
                - Inner list contains images for the task
                - In the last row, condition images are provided and the target images are placed as None
            vae_scale_factor (int):
                The scale factor used by the VAE for resizing images
