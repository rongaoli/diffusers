import os
import tempfile
from typing import Any, Callable, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

import PIL.Image
import PIL.ImageOps
import requests

from .constants import DIFFUSERS_REQUEST_TIMEOUT
from .import_utils import BACKENDS_MAPPING, is_imageio_available

def load_image(
    image: Union[str, PIL.Image.Image], convert_method: Optional[Callable[[PIL.Image.Image], PIL.Image.Image]] = None
) -> PIL.Image.Image:

    Loads `video` to a list of PIL Image.
        `List[PIL.Image.Image]`:
            The video as a list of PIL images.
    is_url = video.startswith("http://") or video.startswith("https://")
    is_file = os.path.isfile(video)
    was_tempfile_created = False

    if not (is_url or is_file):
        raise ValueError(
            f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {video} is not a valid path."
        )

    if is_url:
        response = requests.get(video, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Failed to download video. Status code: {response.status_code}")

        parsed_url = urlparse(video)
        file_name = os.path.basename(unquote(parsed_url.path))

        suffix = os.path.splitext(file_name)[1] or ".mp4"
        video_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name

        was_tempfile_created = True

        video_data = response.iter_content(chunk_size=8192)
        with open(video_path, "wb") as f:
            for chunk in video_data:
                f.write(chunk)

        video = video_path

    pil_images = []
    if video.endswith(".gif"):
        gif = PIL.Image.open(video)
        try:
            while True:
                pil_images.append(gif.copy())
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

    else:
        if is_imageio_available():
            import imageio
        else:
            raise ImportError(BACKENDS_MAPPING["imageio"][1].format("load_video"))

        try:
            imageio.plugins.ffmpeg.get_exe()
        except AttributeError:
            raise AttributeError(
                "`Unable to find an ffmpeg installation on your machine. Please install via `pip install imageio-ffmpeg"
            )

        with imageio.get_reader(video) as reader:
            # Read all frames
            for frame in reader:
                pil_images.append(PIL.Image.fromarray(frame))

    if was_tempfile_created:
        os.remove(video_path)

    if convert_method is not None:
        pil_images = convert_method(pil_images)

    return pil_images

# Taken from `transformers`.
def get_module_from_name(module, tensor_name: str) -> Tuple[Any, str]:
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]
    return module, tensor_name

def get_submodule_by_name(root_module, module_path: str):
    current = root_module
    parts = module_path.split(".")
    for part in parts:
        if part.isdigit():
            idx = int(part)
            current = current[idx]  # e.g., for nn.ModuleList or nn.Sequential
        else:
            current = getattr(current, part)
    return current
