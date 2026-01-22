# coding=utf-8
import io
import json
from typing import List, Literal, Optional, Union, cast

import requests

from .deprecation_utils import deprecate
from .import_utils import is_safetensors_available, is_torch_available

if is_torch_available():
    import torch

    from ..image_processor import VaeImageProcessor
    from ..video_processor import VideoProcessor

    if is_safetensors_available():
        import safetensors.torch

    DTYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
    }

from PIL import Image

def detect_image_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8"):
        return "jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    elif data.startswith(b"BM"):
        return "bmp"
    return "unknown"

def check_inputs_decode(
    endpoint: str,
    tensor: "torch.Tensor",
    processor: Optional[Union["VaeImageProcessor", "VideoProcessor"]] = None,
    do_scaling: bool = True,
    scaling_factor: Optional[float] = None,
    shift_factor: Optional[float] = None,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    return_type: Literal["mp4", "pil", "pt"] = "pil",
    image_format: Literal["png", "jpg"] = "jpg",
    partial_postprocess: bool = False,
    input_tensor_type: Literal["binary"] = "binary",
    output_tensor_type: Literal["binary"] = "binary",
    height: Optional[int] = None,
    width: Optional[int] = None,
):
    if tensor.ndim == 3 and height is None and width is None:
        raise ValueError("`height` and `width` required for packed latents.")
    if (
        output_type == "pt"
        and return_type == "pil"
        and not partial_postprocess
        and not isinstance(processor, (VaeImageProcessor, VideoProcessor))
    ):
        raise ValueError("`processor` is required.")
    if do_scaling and scaling_factor is None:
        deprecate(
            "do_scaling",
            "1.0.0",
            "`do_scaling` is deprecated, pass `scaling_factor` and `shift_factor` if required.",
            standard_warn=False,
        )

def postprocess_decode(
    response: requests.Response,
    processor: Optional[Union["VaeImageProcessor", "VideoProcessor"]] = None,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    return_type: Literal["mp4", "pil", "pt"] = "pil",
    partial_postprocess: bool = False,
):
    if output_type == "pt" or (output_type == "pil" and processor is not None):
        output_tensor = response.content
        parameters = response.headers
        shape = json.loads(parameters["shape"])
        dtype = parameters["dtype"]
        torch_dtype = DTYPE_MAP[dtype]
        output_tensor = torch.frombuffer(bytearray(output_tensor), dtype=torch_dtype).reshape(shape)
    if output_type == "pt":
        if partial_postprocess:
            if return_type == "pil":
                output = [Image.fromarray(image.numpy()) for image in output_tensor]
                if len(output) == 1:
                    output = output[0]
            elif return_type == "pt":
                output = output_tensor
        else:
            if processor is None or return_type == "pt":
                output = output_tensor
            else:
                if isinstance(processor, VideoProcessor):
                    output = cast(
                        List[Image.Image],
                        processor.postprocess_video(output_tensor, output_type="pil")[0],
                    )
                else:
                    output = cast(
                        Image.Image,
                        processor.postprocess(output_tensor, output_type="pil")[0],
                    )
    elif output_type == "pil" and return_type == "pil" and processor is None:
        output = Image.open(io.BytesIO(response.content)).convert("RGB")
        detected_format = detect_image_type(response.content)
        output.format = detected_format
    elif output_type == "pil" and processor is not None:
        if return_type == "pil":
            output = [
                Image.fromarray(image)
                for image in (output_tensor.permute(0, 2, 3, 1).float().numpy() * 255).round().astype("uint8")
            ]
        elif return_type == "pt":
            output = output_tensor
    elif output_type == "mp4" and return_type == "mp4":
        output = response.content
    return output

def prepare_decode(
    tensor: "torch.Tensor",
    processor: Optional[Union["VaeImageProcessor", "VideoProcessor"]] = None,
    do_scaling: bool = True,
    scaling_factor: Optional[float] = None,
    shift_factor: Optional[float] = None,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    image_format: Literal["png", "jpg"] = "jpg",
    partial_postprocess: bool = False,
    height: Optional[int] = None,
    width: Optional[int] = None,
):
    headers = {}
    parameters = {
        "image_format": image_format,
        "output_type": output_type,
        "partial_postprocess": partial_postprocess,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).split(".")[-1],
    }
    if do_scaling and scaling_factor is not None:
        parameters["scaling_factor"] = scaling_factor
    if do_scaling and shift_factor is not None:
        parameters["shift_factor"] = shift_factor
    if do_scaling and scaling_factor is None:
        parameters["do_scaling"] = do_scaling
    elif do_scaling and scaling_factor is None and shift_factor is None:
        parameters["do_scaling"] = do_scaling
    if height is not None and width is not None:
        parameters["height"] = height
        parameters["width"] = width
    headers["Content-Type"] = "tensor/binary"
    headers["Accept"] = "tensor/binary"
    if output_type == "pil" and image_format == "jpg" and processor is None:
        headers["Accept"] = "image/jpeg"
    elif output_type == "pil" and image_format == "png" and processor is None:
        headers["Accept"] = "image/png"
    elif output_type == "mp4":
        headers["Accept"] = "text/plain"
    tensor_data = safetensors.torch._tobytes(tensor, "tensor")
    return {"data": tensor_data, "params": parameters, "headers": headers}

def remote_decode(
    endpoint: str,
    tensor: "torch.Tensor",
    processor: Optional[Union["VaeImageProcessor", "VideoProcessor"]] = None,
    do_scaling: bool = True,
    scaling_factor: Optional[float] = None,
    shift_factor: Optional[float] = None,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    return_type: Literal["mp4", "pil", "pt"] = "pil",
    image_format: Literal["png", "jpg"] = "jpg",
    partial_postprocess: bool = False,
    input_tensor_type: Literal["binary"] = "binary",
    output_tensor_type: Literal["binary"] = "binary",
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Union[Image.Image, List[Image.Image], bytes, "torch.Tensor"]:

    Hugging Face Hybrid Inference that allow running VAE encode remotely.
            - SD v1: 0.18215
            - SD XL: 0.13025
            - Flux: 0.3611
            If `None`, input must be passed with scaling applied.
        shift_factor (`float`, *optional*):
            Shift is applied when passed e.g. `latents - self.vae.config.shift_factor`.
            - Flux: 0.1159
            If `None`, input must be passed with scaling applied.
