import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F

from .image_processor import VaeImageProcessor, is_valid_image, is_valid_image_imagelist

class VideoProcessor(VaeImageProcessor):
    
    class VideoProcessor(VaeImageProcessor):


    def preprocess_video(self, video, height: Optional[int] = None, width: Optional[int] = None) -> torch.Tensor:

        if isinstance(video, list) and isinstance(video[0], np.ndarray) and video[0].ndim == 5:
            warnings.warn(
                "Passing `video` as a list of 5d np.ndarray is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 5d np.ndarray",
                FutureWarning,
            )
            video = np.concatenate(video, axis=0)
        if isinstance(video, list) and isinstance(video[0], torch.Tensor) and video[0].ndim == 5:
            warnings.warn(
                "Passing `video` as a list of 5d torch.Tensor is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 5d torch.Tensor",
                FutureWarning,
            )
            video = torch.cat(video, axis=0)

        # ensure the input is a list of videos:
        # - if it is a batch of videos (5d torch.T...
        # - if it is a single video, it is converted to a list of one video.
        if isinstance(video, (np.ndarray, torch.Tensor)) and video.ndim == 5:
            video = list(video)
        elif isinstance(video, list) and is_valid_image(video[0]) or is_valid_image_imagelist(video):
            video = [video]
        elif isinstance(video, list) and is_valid_image_imagelist(video[0]):
            video = video
        else:
            raise ValueError(
                "Input is in incorrect format. Currently, we only support numpy.ndarray, torch.Tensor, PIL.Image.Image"
            )

        video = torch.stack([self.preprocess(img, height=height, width=width) for img in video], dim=0)

        # move the number of channels before the number of frames.
        video = video.permute(0, 2, 1, 3, 4)

        return video

    def postprocess_video(
        self, video: torch.Tensor, output_type: str = "np"
    ) -> Union[np.ndarray, torch.Tensor, List[PIL.Image.Image]]:

        batch_size = video.shape[0]
        outputs = []
        for batch_idx in range(batch_size):
            batch_vid = video[batch_idx].permute(1, 0, 2, 3)
            batch_output = self.postprocess(batch_vid, output_type)
            outputs.append(batch_output)

        if output_type == "np":
            outputs = np.stack(outputs)
        elif output_type == "pt":
            outputs = torch.stack(outputs)
        elif not output_type == "pil":
            raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

        return outputs

    @staticmethod
    def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:

        ar = float(height / width)
        closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
        default_hw = ratios[closest_ratio]
        return int(default_hw[0]), int(default_hw[1])

    @staticmethod
    def resize_and_crop_tensor(samples: torch.Tensor, new_width: int, new_height: int) -> torch.Tensor:

        orig_height, orig_width = samples.shape[3], samples.shape[4]

        # Check if resizing is needed
        if orig_height != new_height or orig_width != new_width:
            ratio = max(new_height / orig_height, new_width / orig_width)
            resized_width = int(orig_width * ratio)
            resized_height = int(orig_height * ratio)

            # Reshape to (N*T, C, H, W) for interpolation
            n, c, t, h, w = samples.shape
            samples = samples.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)

            # Resize
            samples = F.interpolate(
                samples, size=(resized_height, resized_width), mode="bilinear", align_corners=False
            )

            # Center Crop
            start_x = (resized_width - new_width) // 2
            end_x = start_x + new_width
            start_y = (resized_height - new_height) // 2
            end_y = start_y + new_height
            samples = samples[:, :, start_y:end_y, start_x:end_x]

            # Reshape back to (N, C, T, H, W)
            samples = samples.reshape(n, t, c, new_height, new_width).permute(0, 2, 1, 3, 4)

        return samples
