from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from ....models import UNet2DModel
from ....schedulers import RePaintScheduler
from ....utils import PIL_INTERPOLATION, deprecate, logging
from ....utils.torch_utils import randn_tensor
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def _preprocess_image(image: Union[List, PIL.Image.Image, torch.Tensor]):
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

def _preprocess_mask(mask: Union[List, PIL.Image.Image, torch.Tensor]):
    if isinstance(mask, torch.Tensor):
        return mask
    elif isinstance(mask, PIL.Image.Image):
        mask = [mask]

    if isinstance(mask[0], PIL.Image.Image):
        w, h = mask[0].size
        w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
        mask = [np.array(m.convert("L").resize((w, h), resample=PIL_INTERPOLATION["nearest"]))[None, :] for m in mask]
        mask = np.concatenate(mask, axis=0)
        mask = mask.astype(np.float32) / 255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
    elif isinstance(mask[0], torch.Tensor):
        mask = torch.cat(mask, dim=0)
    return mask

class RePaintPipeline(DiffusionPipeline):


    unet: UNet2DModel
    scheduler: RePaintScheduler
    model_cpu_offload_seq = "unet"

    def __init__(self, unet: UNet2DModel, scheduler: RePaintScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, PIL.Image.Image],
        mask_image: Union[torch.Tensor, PIL.Image.Image],
        num_inference_steps: int = 250,
        eta: float = 0.0,
        jump_length: int = 10,
        jump_n_sample: int = 10,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        """r"""

        使用示例见文档

