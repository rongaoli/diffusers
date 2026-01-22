import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

from .configuration_utils import ConfigMixin, register_to_config
from .utils import CONFIG_NAME, PIL_INTERPOLATION, deprecate

PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]

PipelineDepthInput = PipelineImageInput

def is_valid_image(image) -> bool:

    return isinstance(image, PIL.Image.Image) or isinstance(image, (np.ndarray, torch.Tensor)) and image.ndim in (2, 3)

def is_valid_image_imagelist(images):

    if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4:
        return True
    elif is_valid_image(images):
        return True
    elif isinstance(images, list):
        return all(is_valid_image(image) for image in images)
    return False

class VaeImageProcessor(ConfigMixin):
    
    class VaeImageProcessor(ConfigMixin):


    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        vae_latent_channels: int = 4,
        resample: str = "lanczos",
        reducing_gap: int = None,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
    ):
        super().__init__()
        if do_convert_rgb and do_convert_grayscale:
            raise ValueError(
                "`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`,"
                " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`.",
                " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`",
            )

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:

        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @staticmethod
    def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:

        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)

        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:

        if images.ndim == 3:
            images = images[..., None]

        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:

        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def normalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:

        return 2.0 * images - 1.0

    @staticmethod
    def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:

        return (images * 0.5 + 0.5).clamp(0, 1)

    @staticmethod
    def convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:

        image = image.convert("RGB")

        return image

    @staticmethod
    def convert_to_grayscale(image: PIL.Image.Image) -> PIL.Image.Image:

        image = image.convert("L")

        return image

    @staticmethod
    def blur(image: PIL.Image.Image, blur_factor: int = 4) -> PIL.Image.Image:

        image = image.filter(ImageFilter.GaussianBlur(blur_factor))

        return image

    @staticmethod
    def get_crop_region(mask_image: PIL.Image.Image, width: int, height: int, pad=0):


        mask_image = mask_image.convert("L")
        mask = np.array(mask_image)

        # 1. find a rectangular region that contains all masked ares in an image
        h, w = mask.shape
        crop_left = 0
        for i in range(w):
            if not (mask[:, i] == 0).all():
                break
            crop_left += 1

        crop_right = 0
        for i in reversed(range(w)):
            if not (mask[:, i] == 0).all():
                break
            crop_right += 1

        crop_top = 0
        for i in range(h):
            if not (mask[i] == 0).all():
                break
            crop_top += 1

        crop_bottom = 0
        for i in reversed(range(h)):
            if not (mask[i] == 0).all():
                break
            crop_bottom += 1

        # 2. add padding to the crop region
        x1, y1, x2, y2 = (
            int(max(crop_left - pad, 0)),
            int(max(crop_top - pad, 0)),
            int(min(w - crop_right + pad, w)),
            int(min(h - crop_bottom + pad, h)),
        )

        # 3. expands crop region to match the aspect ratio of the image to be processed
        ratio_crop_region = (x2 - x1) / (y2 - y1)
        ratio_processing = width / height

        if ratio_crop_region > ratio_processing:
            desired_height = (x2 - x1) / ratio_processing
            desired_height_diff = int(desired_height - (y2 - y1))
            y1 -= desired_height_diff // 2
            y2 += desired_height_diff - desired_height_diff // 2
            if y2 >= mask_image.height:
                diff = y2 - mask_image.height
                y2 -= diff
                y1 -= diff
            if y1 < 0:
                y2 -= y1
                y1 -= y1
            if y2 >= mask_image.height:
                y2 = mask_image.height
        else:
            desired_width = (y2 - y1) * ratio_processing
            desired_width_diff = int(desired_width - (x2 - x1))
            x1 -= desired_width_diff // 2
            x2 += desired_width_diff - desired_width_diff // 2
            if x2 >= mask_image.width:
                diff = x2 - mask_image.width
                x2 -= diff
                x1 -= diff
            if x1 < 0:
                x2 -= x1
                x1 -= x1
            if x2 >= mask_image.width:
                x2 = mask_image.width

        return x1, y1, x2, y2

    def _resize_and_fill(
        self,
        image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:


        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio < src_ratio else image.width * height // image.height
        src_h = height if ratio >= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION[self.config.resample])
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

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

    def _resize_and_crop(
        self,
        image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:

        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio > src_ratio else image.width * height // image.height
        src_h = height if ratio <= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION[self.config.resample])
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        return res

    def resize(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        height: int,
        width: int,
        resize_mode: str = "default",  # "default", "fill", "crop"
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:

        if resize_mode != "default" and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Only PIL image input is supported for resize_mode {resize_mode}")
        if isinstance(image, PIL.Image.Image):
            if resize_mode == "default":
                image = image.resize(
                    (width, height),
                    resample=PIL_INTERPOLATION[self.config.resample],
                    reducing_gap=self.config.reducing_gap,
                )
            elif resize_mode == "fill":
                image = self._resize_and_fill(image, width, height)
            elif resize_mode == "crop":
                image = self._resize_and_crop(image, width, height)
            else:
                raise ValueError(f"resize_mode {resize_mode} is not supported")

        elif isinstance(image, torch.Tensor):
            image = torch.nn.functional.interpolate(
                image,
                size=(height, width),
            )
        elif isinstance(image, np.ndarray):
            image = self.numpy_to_pt(image)
            image = torch.nn.functional.interpolate(
                image,
                size=(height, width),
            )
            image = self.pt_to_numpy(image)

        return image

    def binarize(self, image: PIL.Image.Image) -> PIL.Image.Image:

        image[image < 0.5] = 0
        image[image >= 0.5] = 1

        return image

    def _denormalize_conditionally(
        self, images: torch.Tensor, do_denormalize: Optional[List[bool]] = None
    ) -> torch.Tensor:

        if do_denormalize is None:
            return self.denormalize(images) if self.config.do_normalize else images

        return torch.stack(
            [self.denormalize(images[i]) if do_denormalize[i] else images[i] for i in range(images.shape[0])]
        )

    def get_default_height_width(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[int, int]:


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

        width, height = (
            x - x % self.config.vae_scale_factor for x in (width, height)
        )  # resize to integer multiple of vae_scale_factor

        return height, width

    def preprocess(
        self,
        image: PipelineImageInput,
        height: Optional[int] = None,
        width: Optional[int] = None,
        resize_mode: str = "default",  # "default", "fill", "crop"
        crops_coords: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:

        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

        # Expand the missing dimension for 3-dimen...
        if self.config.do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
            if isinstance(image, torch.Tensor):
                # if image is a pytorch tensor could have 2 possible shapes:
                #    1. batch x height x width: we should insert the channel dimension at position 1
                #    2. channel x height x width: we should insert batch dimension at position 0,
                #       however, since both channe...
                #    for simplicity, we insert a dimension of size 1 at position 1 for both cases
                image = image.unsqueeze(1)
            else:
                # if it is a numpy array, it could have 2 possible shapes:
                #   1. batch x height x width: insert channel dimension on last position
                #   2. height x width x channel: insert batch dimension on first position
                if image.shape[-1] == 1:
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=-1)

        if isinstance(image, list) and isinstance(image[0], np.ndarray) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d np.ndarray is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d np.ndarray",
                FutureWarning,
            )
            image = np.concatenate(image, axis=0)
        if isinstance(image, list) and isinstance(image[0], torch.Tensor) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d torch.Tensor is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d torch.Tensor",
                FutureWarning,
            )
            image = torch.cat(image, axis=0)

        if not is_valid_image_imagelist(image):
            raise ValueError(
                f"Input is in incorrect format. Currently, we only support {', '.join(str(x) for x in supported_formats)}"
            )
        if not isinstance(image, list):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            if crops_coords is not None:
                image = [i.crop(crops_coords) for i in image]
            if self.config.do_resize:
                height, width = self.get_default_height_width(image[0], height, width)
                image = [self.resize(i, height, width, resize_mode=resize_mode) for i in image]
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            elif self.config.do_convert_grayscale:
                image = [self.convert_to_grayscale(i) for i in image]
            image = self.pil_to_numpy(image)  # to np
            image = self.numpy_to_pt(image)  # to pt

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)

            image = self.numpy_to_pt(image)

            height, width = self.get_default_height_width(image, height, width)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

            if self.config.do_convert_grayscale and image.ndim == 3:
                image = image.unsqueeze(1)

            channel = image.shape[1]
            # don't need any preprocess if the image is latents
            if channel == self.config.vae_latent_channels:
                return image

            height, width = self.get_default_height_width(image, height, width)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize
        if do_normalize and image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False
        if do_normalize:
            image = self.normalize(image)

        if self.config.do_binarize:
            image = self.binarize(image)

        return image

    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        Postprocess the image output from tensor to `output_type`.
                `VaeImageProcessor` config.
