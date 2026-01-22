import os
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from .modeling_utils import ModelMixin

logger = logging.get_logger(__name__)

class MultiAdapter(ModelMixin):


    def __init__(self, adapters: List["T2IAdapter"]):
        super(MultiAdapter, self).__init__()

        self.num_adapter = len(adapters)
        self.adapters = nn.ModuleList(adapters)

        if len(adapters) == 0:
            raise ValueError("Expecting at least one adapter")

        if len(adapters) == 1:
            raise ValueError("For a single adapter, please use the `T2IAdapter` class instead of `MultiAdapter`")

        # The outputs from each adapter are added together with a weight.
        # This means that the change in dimensions from downsampling must
        # be the same for all adapters. Inductively, it also means the
        # downscale_factor and total_downscale_factor must be the same for all
        # adapters.
        first_adapter_total_downscale_factor = adapters[0].total_downscale_factor
        first_adapter_downscale_factor = adapters[0].downscale_factor
        for idx in range(1, len(adapters)):
            if (
                adapters[idx].total_downscale_factor != first_adapter_total_downscale_factor
                or adapters[idx].downscale_factor != first_adapter_downscale_factor
            ):
                raise ValueError(
                    f"Expecting all adapters to have the same downscaling behavior, but got:\n"
                    f"adapters[0].total_downscale_factor={first_adapter_total_downscale_factor}\n"
                    f"adapters[0].downscale_factor={first_adapter_downscale_factor}\n"
                    f"adapter[`{idx}`].total_downscale_factor={adapters[idx].total_downscale_factor}\n"
                    f"adapter[`{idx}`].downscale_factor={adapters[idx].downscale_factor}"
                )

        self.total_downscale_factor = first_adapter_total_downscale_factor
        self.downscale_factor = first_adapter_downscale_factor

    def forward(self, xs: torch.Tensor, adapter_weights: Optional[List[float]] = None) -> List[torch.Tensor]:
        class instead of `MultiAdapter`")

        # The outputs from each adapter are added together with a weight.
        # This means that the change in dimensions from downsampling must
        # be the same for all adapters. Inductively, it also means the
        # downscale_factor and total_downscale_factor must be the same for all
        # adapters.
        first_adapter_total_downscale_factor = adapters[0].total_downscale_factor
        first_adapter_downscale_factor = adapters[0].downscale_factor
        for idx in range(1, len(adapters)):
            if (
                adapters[idx].total_downscale_factor != first_adapter_total_downscale_factor
                or adapters[idx].downscale_factor != first_adapter_downscale_factor
            ):
                raise ValueError(
                    f"Expecting all adapters to have the same downscaling behavior, but got:\n"
                    f"adapters[0].total_downscale_factor={first_adapter_total_downscale_factor}\n"
                    f"adapters[0].downscale_factor={first_adapter_downscale_factor}\n"
                    f"adapter[`{idx}`].total_downscale_factor={adapters[idx].total_downscale_factor}\n"
                    f"adapter[`{idx}`].downscale_factor={adapters[idx].downscale_factor}"
                )

        self.total_downscale_factor = first_adapter_total_downscale_factor
        self.downscale_factor = first_adapter_downscale_factor

    def forward(self, xs: torch.Tensor, adapter_weights: Optional[List[float]] = None) -> List[torch.Tensor]:

        if adapter_weights is None:
            adapter_weights = torch.tensor([1 / self.num_adapter] * self.num_adapter)
        else:
            adapter_weights = torch.tensor(adapter_weights)

        accume_state = None
        for x, w, adapter in zip(xs, adapter_weights, self.adapters):
            features = adapter(x)
            if accume_state is None:
                accume_state = features
                for i in range(len(accume_state)):
                    accume_state[i] = w * accume_state[i]
            else:
                for i in range(len(features)):
                    accume_state[i] += w * features[i]
        return accume_state

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
    ):

        idx = 0
        model_path_to_save = save_directory
        for adapter in self.adapters:
            adapter.save_pretrained(
                model_path_to_save,
                is_main_process=is_main_process,
                save_function=save_function,
                safe_serialization=safe_serialization,
                variant=variant,
            )

            idx += 1
            model_path_to_save = model_path_to_save + f"_{idx}"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Optional[Union[str, os.PathLike]], **kwargs):
        class method.
                `torch.save` with another method. Can also be configured using`DIFFUSERS_SAVE_MODE` environment
                variable.
            safe_serialization (`bool`, optional, defaults=True):
                If `True`, save the model using `safetensors`. If `False`, save the model with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.

        idx = 0
        adapters = []

        # load adapter and append to list until no adapter directory exists anymore
        # first adapter has to be saved under `./m...
        # second, third, ... adapters have to be s...
        model_path_to_load = pretrained_model_path
        while os.path.isdir(model_path_to_load):
            adapter = T2IAdapter.from_pretrained(model_path_to_load, **kwargs)
            adapters.append(adapter)

            idx += 1
            model_path_to_load = pretrained_model_path + f"_{idx}"

        logger.info(f"{len(adapters)} adapters loaded from {pretrained_model_path}.")

        if len(adapters) == 0:
            raise ValueError(
                f"No T2IAdapters found under {os.path.dirname(pretrained_model_path)}. Expected at least {pretrained_model_path + '_0'}."
            )

        return cls(adapters)

class T2IAdapter(ModelMixin, ConfigMixin):


    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
        adapter_type: str = "full_adapter",
    ):
        super().__init__()

        if adapter_type == "full_adapter":
            self.adapter = FullAdapter(in_channels, channels, num_res_blocks, downscale_factor)
        elif adapter_type == "full_adapter_xl":
            self.adapter = FullAdapterXL(in_channels, channels, num_res_blocks, downscale_factor)
        elif adapter_type == "light_adapter":
            self.adapter = LightAdapter(in_channels, channels, num_res_blocks, downscale_factor)
        else:
            raise ValueError(
                f"Unsupported adapter_type: '{adapter_type}'. Choose either 'full_adapter' or "
                "'full_adapter_xl' or 'light_adapter'."
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        
        """r"""
        return self.adapter(x)

    @property
    def total_downscale_factor(self):
        return self.adapter.total_downscale_factor

    @property
    def downscale_factor(self):

        return self.adapter.unshuffle.downscale_factor

# full adapter

class FullAdapter(nn.Module):


    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.body = nn.ModuleList(
            [
                AdapterBlock(channels[0], channels[0], num_res_blocks),
                *[
                    AdapterBlock(channels[i - 1], channels[i], num_res_blocks, down=True)
                    for i in range(1, len(channels))
                ],
            ]
        )

        self.total_downscale_factor = downscale_factor * 2 ** (len(channels) - 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        
        """r"""
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features

class FullAdapterXL(nn.Module):


    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 16,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.body = []
        # blocks to extract XL features with dimen...
        for i in range(len(channels)):
            if i == 1:
                self.body.append(AdapterBlock(channels[i - 1], channels[i], num_res_blocks))
            elif i == 2:
                self.body.append(AdapterBlock(channels[i - 1], channels[i], num_res_blocks, down=True))
            else:
                self.body.append(AdapterBlock(channels[i], channels[i], num_res_blocks))

        self.body = nn.ModuleList(self.body)
        # XL has only one downsampling AdapterBlock.
        self.total_downscale_factor = downscale_factor * 2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        
        """r"""
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features

class AdapterBlock(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False):
        super().__init__()

        self.downsample = None
        if down:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.in_conv = None
        if in_channels != out_channels:
            self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.resnets = nn.Sequential(
            *[AdapterResnetBlock(out_channels) for _ in range(num_res_blocks)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """r"""
        if self.downsample is not None:
            x = self.downsample(x)

        if self.in_conv is not None:
            x = self.in_conv(x)

        x = self.resnets(x)

        return x

class AdapterResnetBlock(nn.Module):


    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """r"""

        h = self.act(self.block1(x))
        h = self.block2(h)

        return h + x

# light adapter

class LightAdapter(nn.Module):


    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280],
        num_res_blocks: int = 4,
        downscale_factor: int = 8,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)

        self.body = nn.ModuleList(
            [
                LightAdapterBlock(in_channels, channels[0], num_res_blocks),
                *[
                    LightAdapterBlock(channels[i], channels[i + 1], num_res_blocks, down=True)
                    for i in range(len(channels) - 1)
                ],
                LightAdapterBlock(channels[-1], channels[-1], num_res_blocks, down=True),
            ]
        )

        self.total_downscale_factor = downscale_factor * (2 ** len(channels))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        
        """r"""
        x = self.unshuffle(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features

class LightAdapterBlock(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False):
        super().__init__()
        mid_channels = out_channels // 4

        self.downsample = None
        if down:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.resnets = nn.Sequential(*[LightAdapterResnetBlock(mid_channels) for _ in range(num_res_blocks)])
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """r"""
        if self.downsample is not None:
            x = self.downsample(x)

        x = self.in_conv(x)
        x = self.resnets(x)
        x = self.out_conv(x)

        return x

class LightAdapterResnetBlock(nn.Module):


    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """r"""

        h = self.act(self.block1(x))
        h = self.block2(h)

        return h + x
