import html
import inspect
import re
import urllib.parse as ul
from typing import Callable, Dict, List, Optional, Union

import ftfy
import torch
from transformers import (
    AutoTokenizer,
    GemmaTokenizerFast,
    T5TokenizerFast,
)
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder

from diffusers.image_processor import PixArtImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderDC, AutoencoderKL
from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.prx.pipeline_output import PRXPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor

DEFAULT_RESOLUTION = 512

ASPECT_RATIO_256_BIN = {
    "0.46": [160, 352],
    "0.6": [192, 320],
    "0.78": [224, 288],
    "1.0": [256, 256],
    "1.29": [288, 224],
    "1.67": [320, 192],
    "2.2": [352, 160],
}

ASPECT_RATIO_512_BIN = {
    "0.5": [352, 704],
    "0.57": [384, 672],
    "0.6": [384, 640],
    "0.68": [416, 608],
    "0.78": [448, 576],
    "0.88": [480, 544],
    "1.0": [512, 512],
    "1.13": [544, 480],
    "1.29": [576, 448],
    "1.46": [608, 416],
    "1.67": [640, 384],
    "1.75": [672, 384],
    "2.0": [704, 352],
}

ASPECT_RATIO_1024_BIN = {
    "0.49": [704, 1440],
    "0.52": [736, 1408],
    "0.53": [736, 1376],
    "0.57": [768, 1344],
    "0.59": [768, 1312],
    "0.62": [800, 1280],
    "0.67": [832, 1248],
    "0.68": [832, 1216],
    "0.78": [896, 1152],
    "0.83": [928, 1120],
    "0.94": [992, 1056],
    "1.0": [1024, 1024],
    "1.06": [1056, 992],
    "1.13": [1088, 960],
    "1.21": [1120, 928],
    "1.29": [1152, 896],
    "1.37": [1184, 864],
    "1.46": [1216, 832],
    "1.5": [1248, 832],
    "1.71": [1312, 768],
    "1.75": [1344, 768],
    "1.87": [1376, 736],
    "1.91": [1408, 736],
    "2.05": [1440, 704],
}

ASPECT_RATIO_BINS = {
    256: ASPECT_RATIO_256_BIN,
    512: ASPECT_RATIO_512_BIN,
    1024: ASPECT_RATIO_1024_BIN,
}

logger = logging.get_logger(__name__)

class TextPreprocessor:


    def __init__(self):

        self.bad_punct_regex = re.compile(
            r"["
            + "#®•©™&@·º½¾¿¡§~"
            + r"\)"
            + r"\("
            + r"\]"
            + r"\["
            + r"\}"
            + r"\{"
            + r"\|"
            + r"\\"
            + r"\/"
            + r"\*"
            + r"]{1,}"
        )

    def clean_text(self, text: str) -> str:

        # See Deepfloyd https://github.com/deep-floyd/IF/blob/develop/deepfloyd_if/modules/t5.py
        text = str(text)
        text = ul.unquote_plus(text)
        text = text.strip().lower()
        text = re.sub("<person>", "person", text)

        # Remove all urls:
        text = re.sub(
            r"\b((?:https?|www):(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@))",
            "",
            text,
        )  # regex for urls

        # @<nickname>
        text = re.sub(r"@[\w\d]+\b", "", text)

        # 31C0—31EF CJK Strokes through 4E00—9FFF CJK Unified Ideographs
        text = re.sub(r"[\u31c0-\u31ef]+", "", text)
        text = re.sub(r"[\u31f0-\u31ff]+", "", text)
        text = re.sub(r"[\u3200-\u32ff]+", "", text)
        text = re.sub(r"[\u3300-\u33ff]+", "", text)
        text = re.sub(r"[\u3400-\u4dbf]+", "", text)
        text = re.sub(r"[\u4dc0-\u4dff]+", "", text)
        text = re.sub(r"[\u4e00-\u9fff]+", "", text)

        # все виды тире / all types of dash --> "-"
        text = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            "-",
            text,
        )

        # кавычки к одному стандарту
        text = re.sub(r"[`´«»" "¨]", '"', text)
        text = re.sub(r"['']", "'", text)

        # &quot; and &amp
        text = re.sub(r"&quot;?", "", text)
        text = re.sub(r"&amp", "", text)

        # ip addresses:
        text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", text)

        # article ids:
        text = re.sub(r"\d:\d\d\s+$", "", text)

        # \n
        text = re.sub(r"\\n", " ", text)

        # "#123", "#12345..", "123456.."
        text = re.sub(r"#\d{1,3}\b", "", text)
        text = re.sub(r"#\d{5,}\b", "", text)
        text = re.sub(r"\b\d{6,}\b", "", text)

        # filenames:
        text = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", text)

        # Clean punctuation
        text = re.sub(r"[\"\']{2,}", r'"', text)  # """AUSVERKAUFT"""
        text = re.sub(r"[\.]{2,}", r" ", text)

        text = re.sub(self.bad_punct_regex, r" ", text)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        text = re.sub(r"\s+\.\s+", r" ", text)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, text)) > 3:
            text = re.sub(regex2, " ", text)

        # Basic cleaning
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = text.strip()

        # Clean alphanumeric patterns
        text = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", text)  # jc6640
        text = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", text)  # jc6640vc
        text = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", text)  # 6640vc231

        # Common spam patterns
        text = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", text)
        text = re.sub(r"(free\s)?download(\sfree)?", "", text)
        text = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", text)
        text = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", text)
        text = re.sub(r"\bpage\s+\d+\b", "", text)

        text = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", text)  # j2d1a2a...
        text = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", text)

        # Final cleanup
        text = re.sub(r"\b\s+\:\s+", r": ", text)
        text = re.sub(r"(\D[,\./])\b", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)

        text.strip()

        text = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", text)
        text = re.sub(r"^[\'\_,\-\:;]", r"", text)
        text = re.sub(r"[\'\_,\-\:\-\+]$", r"", text)
        text = re.sub(r"^\.\S+$", "", text)

        return text.strip()

EXAMPLE_DOC_STRING = """
        使用示例见文档

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer: PRXTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: T5GemmaEncoder,
        tokenizer: Union[T5TokenizerFast, GemmaTokenizerFast, AutoTokenizer],
        vae: Optional[Union[AutoencoderKL, AutoencoderDC]] = None,
        default_sample_size: Optional[int] = DEFAULT_RESOLUTION,
    ):
        super().__init__()

        if PRXTransformer2DModel is None:
            raise ImportError(
                "PRXTransformer2DModel is not available. Please ensure the transformer_prx module is properly installed."
            )

        self.text_preprocessor = TextPreprocessor()
        self.default_sample_size = default_sample_size
        self._guidance_scale = 1.0

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
        )

        self.register_to_config(default_sample_size=self.default_sample_size)

        if vae is not None:
            self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)
        else:
            self.image_processor = None

    @property
    def vae_scale_factor(self):
        if self.vae is None:
            return 8
        if hasattr(self.vae, "spatial_compression_ratio"):
            return self.vae.spatial_compression_ratio
        else:  # Flux VAE
            return 2 ** (len(self.vae.config.block_out_channels) - 1)

    @property
    def do_classifier_free_guidance(self):

        return self._guidance_scale > 1.0

    @property
    def guidance_scale(self):
        return self._guidance_scale

    def get_default_resolution(self):
        """Determine the default resolution based on the loaded VAE and config.
