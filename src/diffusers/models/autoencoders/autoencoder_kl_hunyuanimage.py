from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class HunyuanImageResnetBlock(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, non_linearity: str = "silu") -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(non_linearity)

        # layers
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def forward(self, x):
        # Apply shortcut connection
        residual = x

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        # Add residual connection
        return x + residual

class HunyuanImageAttentionBlock(nn.Module):


    def __init__(self, in_channels: int):
        super().__init__()

        # layers
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.to_q = nn.Conv2d(in_channels, in_channels, 1)
        self.to_k = nn.Conv2d(in_channels, in_channels, 1)
        self.to_v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        identity = x
        x = self.norm(x)

        # compute query, key, value
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        batch_size, channels, height, width = query.shape
        query = query.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels).contiguous()
        key = key.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels).contiguous()
        value = value.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels).contiguous()

        # apply attention
        x = F.scaled_dot_product_attention(query, key, value)

        x = x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        # output projection
        x = self.proj(x)

        return x + identity

class HunyuanImageDownsample(nn.Module):


    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        factor = 4
        if out_channels % factor != 0:
            raise ValueError(f"out_channels % factor != 0: {out_channels % factor}")

        self.conv = nn.Conv2d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)

        B, C, H, W = h.shape
        h = h.reshape(B, C, H // 2, 2, W // 2, 2)
        h = h.permute(0, 3, 5, 1, 2, 4)  # b, r1, r2, c, h, w
        h = h.reshape(B, 4 * C, H // 2, W // 2)

        B, C, H, W = x.shape
        shortcut = x.reshape(B, C, H // 2, 2, W // 2, 2)
        shortcut = shortcut.permute(0, 3, 5, 1, 2, 4)  # b, r1, r2, c, h, w
        shortcut = shortcut.reshape(B, 4 * C, H // 2, W // 2)

        B, C, H, W = shortcut.shape
        shortcut = shortcut.view(B, h.shape[1], self.group_size, H, W).mean(dim=2)
        return h + shortcut

class HunyuanImageUpsample(nn.Module):


    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        factor = 4
        self.conv = nn.Conv2d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)

        B, C, H, W = h.shape
        h = h.reshape(B, 2, 2, C // 4, H, W)  # b, r1, r2, c, h, w
        h = h.permute(0, 3, 4, 1, 5, 2)  # b, c, h, r1, w, r2
        h = h.reshape(B, C // 4, H * 2, W * 2)

        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)

        B, C, H, W = shortcut.shape
        shortcut = shortcut.reshape(B, 2, 2, C // 4, H, W)  # b, r1, r2, c, h, w
        shortcut = shortcut.permute(0, 3, 4, 1, 5, 2)  # b, c, h, r1, w, r2
        shortcut = shortcut.reshape(B, C // 4, H * 2, W * 2)
        return h + shortcut

class HunyuanImageMidBlock(nn.Module):


    def __init__(self, in_channels: int, num_layers: int = 1):
        super().__init__()

        resnets = [HunyuanImageResnetBlock(in_channels=in_channels, out_channels=in_channels)]

        attentions = []
        for _ in range(num_layers):
            attentions.append(HunyuanImageAttentionBlock(in_channels))
            resnets.append(HunyuanImageResnetBlock(in_channels=in_channels, out_channels=in_channels))

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)

        return x

class HunyuanImageEncoder2D(nn.Module):


    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        spatial_compression_ratio: int,
        non_linearity: str = "silu",
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        if block_out_channels[-1] % (2 * z_channels) != 0:
            raise ValueError(
                f"block_out_channels[-1 has to be divisible by 2 * out_channels, you have block_out_channels = {block_out_channels[-1]} and out_channels = {z_channels}"
            )

        self.in_channels = in_channels
        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks
        self.spatial_compression_ratio = spatial_compression_ratio

        self.group_size = block_out_channels[-1] // (2 * z_channels)
        self.nonlinearity = get_activation(non_linearity)

        # init block
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])

        block_in_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            block_out_channel = block_out_channels[i]
            # residual blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    HunyuanImageResnetBlock(in_channels=block_in_channel, out_channels=block_out_channel)
                )
                block_in_channel = block_out_channel

            # downsample block
            if i < np.log2(spatial_compression_ratio) and i != len(block_out_channels) - 1:
                if downsample_match_channel:
                    block_out_channel = block_out_channels[i + 1]
                self.down_blocks.append(
                    HunyuanImageDownsample(in_channels=block_in_channel, out_channels=block_out_channel)
                )
                block_in_channel = block_out_channel

        # middle blocks
        self.mid_block = HunyuanImageMidBlock(in_channels=block_out_channels[-1], num_layers=1)

        # output blocks
        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out_channels[-1], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_out_channels[-1], 2 * z_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        ## downsamples
        for down_block in self.down_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(down_block, x)
            else:
                x = down_block(x)

        ## middle
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = self._gradient_checkpointing_func(self.mid_block, x)
        else:
            x = self.mid_block(x)

        ## head
        B, C, H, W = x.shape
        residual = x.view(B, C // self.group_size, self.group_size, H, W).mean(dim=2)

        x = self.norm_out(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)
        return x + residual

class HunyuanImageDecoder2D(nn.Module):


    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        spatial_compression_ratio: int,
        upsample_match_channel: bool = True,
        non_linearity: str = "silu",
    ):
        super().__init__()
        if block_out_channels[0] % z_channels != 0:
            raise ValueError(
                f"block_out_channels[0] should be divisible by z_channels but has block_out_channels[0] = {block_out_channels[0]} and z_channels = {z_channels}"
            )

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks
        self.repeat = block_out_channels[0] // z_channels
        self.spatial_compression_ratio = spatial_compression_ratio
        self.nonlinearity = get_activation(non_linearity)

        self.conv_in = nn.Conv2d(z_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        # Middle blocks with attention
        self.mid_block = HunyuanImageMidBlock(in_channels=block_out_channels[0], num_layers=1)

        # Upsampling blocks
        block_in_channel = block_out_channels[0]
        self.up_blocks = nn.ModuleList()
        for i in range(len(block_out_channels)):
            block_out_channel = block_out_channels[i]
            for _ in range(self.num_res_blocks + 1):
                self.up_blocks.append(
                    HunyuanImageResnetBlock(in_channels=block_in_channel, out_channels=block_out_channel)
                )
                block_in_channel = block_out_channel

            if i < np.log2(spatial_compression_ratio) and i != len(block_out_channels) - 1:
                if upsample_match_channel:
                    block_out_channel = block_out_channels[i + 1]
                self.up_blocks.append(HunyuanImageUpsample(block_in_channel, block_out_channel))
                block_in_channel = block_out_channel

        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out_channels[-1], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x) + x.repeat_interleave(repeats=self.repeat, dim=1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            h = self._gradient_checkpointing_func(self.mid_block, h)
        else:
            h = self.mid_block(h)

        for up_block in self.up_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                h = self._gradient_checkpointing_func(up_block, h)
            else:
                h = up_block(h)
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h

class AutoencoderKLHunyuanImage(ModelMixin, AutoencoderMixin, ConfigMixin, FromOriginalModelMixin):


    _supports_gradient_checkpointing = False

    # fmt: off
    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        spatial_compression_ratio: int,
        sample_size: int,
        scaling_factor: float = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
    ) -> None:
    # fmt: on
        super().__init__()

        self.encoder = HunyuanImageEncoder2D(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=layers_per_block,
            spatial_compression_ratio=spatial_compression_ratio,
            downsample_match_channel=downsample_match_channel,
        )

        self.decoder = HunyuanImageDecoder2D(
            z_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(reversed(block_out_channels)),
            num_res_blocks=layers_per_block,
            spatial_compression_ratio=spatial_compression_ratio,
            upsample_match_channel=upsample_match_channel,
        )

        # Tiling and slicing configuration
        self.use_slicing = False
        self.use_tiling = False

        # Tiling parameters
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // spatial_compression_ratio
        self.tile_overlap_factor = 0.25

    def enable_tiling(
        self,
        tile_sample_min_size: Optional[int] = None,
        tile_overlap_factor: Optional[float] = None,
    ) -> None:
        
        """r"""
        self.use_tiling = True
        self.tile_sample_min_size = tile_sample_min_size or self.tile_sample_min_size
        self.tile_overlap_factor = tile_overlap_factor or self.tile_overlap_factor
        self.tile_latent_min_size = self.tile_sample_min_size // self.config.spatial_compression_ratio

    def _encode(self, x: torch.Tensor):

        batch_size, num_channels, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_size or height > self.tile_sample_min_size):
            return self.tiled_encode(x)

        enc = self.encoder(x)

        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        
        """r"""
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True):

        batch_size, num_channels, height, width = z.shape

        if self.use_tiling and (width > self.tile_latent_min_size or height > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        
        """r"""
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:

        _, _, _, height, width = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, height, overlap_size):
            row = []
            for j in range(0, width, overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        moments = torch.cat(result_rows, dim=-2)

        return moments

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:

        _, _, height, width = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, height, overlap_size):
            row = []
            for j in range(0, width, overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=-2)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:

        posterior = self.encode(sample).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)

        return dec
