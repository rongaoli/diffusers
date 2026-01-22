from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import get_timestep_embedding
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm

logger = logging.get_logger(__name__)

def get_image_ids(batch_size: int, height: int, width: int, patch_size: int, device: torch.device) -> torch.Tensor:
    
    """r"""

    img_ids = torch.zeros(height // patch_size, width // patch_size, 2, device=device)
    img_ids[..., 0] = torch.arange(height // patch_size, device=device)[:, None]
    img_ids[..., 1] = torch.arange(width // patch_size, device=device)[None, :]
    return img_ids.reshape((height // patch_size) * (width // patch_size), 2).unsqueeze(0).repeat(batch_size, 1, 1)

def apply_rope(xq: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    
    """r"""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    # Ensure freqs_cis is on the same device as queries to avoid device mismatches with offloading
    freqs_cis = freqs_cis.to(device=xq.device, dtype=xq_.dtype)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)

class PRXAttnProcessor2_0:


    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("PRXAttnProcessor2_0 requires PyTorch 2.0, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: "PRXAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:


        if encoder_hidden_states is None:
            raise ValueError("PRXAttnProcessor2_0 requires 'encoder_hidden_states' containing text tokens.")

        # Project image tokens to Q, K, V
        img_qkv = attn.img_qkv_proj(hidden_states)
        B, L_img, _ = img_qkv.shape
        img_qkv = img_qkv.reshape(B, L_img, 3, attn.heads, attn.head_dim)
        img_qkv = img_qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L_img, D]
        img_q, img_k, img_v = img_qkv[0], img_qkv[1], img_qkv[2]

        # Apply QK normalization to image tokens
        img_q = attn.norm_q(img_q)
        img_k = attn.norm_k(img_k)

        # Project text tokens to K, V
        txt_kv = attn.txt_kv_proj(encoder_hidden_states)
        B, L_txt, _ = txt_kv.shape
        txt_kv = txt_kv.reshape(B, L_txt, 2, attn.heads, attn.head_dim)
        txt_kv = txt_kv.permute(2, 0, 3, 1, 4)  # [2, B, H, L_txt, D]
        txt_k, txt_v = txt_kv[0], txt_kv[1]

        # Apply K normalization to text tokens
        txt_k = attn.norm_added_k(txt_k)

        # Apply RoPE to image queries and keys
        if image_rotary_emb is not None:
            img_q = apply_rope(img_q, image_rotary_emb)
            img_k = apply_rope(img_k, image_rotary_emb)

        # Concatenate text and image keys/values
        k = torch.cat((txt_k, img_k), dim=2)  # [B, H, L_txt + L_img, D]
        v = torch.cat((txt_v, img_v), dim=2)  # [B, H, L_txt + L_img, D]

        # Build attention mask if provided
        attn_mask_tensor = None
        if attention_mask is not None:
            bs, _, l_img, _ = img_q.shape
            l_txt = txt_k.shape[2]

            if attention_mask.dim() != 2:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")
            if attention_mask.shape[-1] != l_txt:
                raise ValueError(f"attention_mask last dim {attention_mask.shape[-1]} must equal text length {l_txt}")

            device = img_q.device
            ones_img = torch.ones((bs, l_img), dtype=torch.bool, device=device)
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
            joint_mask = torch.cat([attention_mask, ones_img], dim=-1)
            attn_mask_tensor = joint_mask[:, None, None, :].expand(-1, attn.heads, l_img, -1)

        # Apply attention using dispatch_attention_fn for backend support
        # Reshape to match dispatch_attention_fn expectations: [B, L, H, D]
        query = img_q.transpose(1, 2)  # [B, L_img, H, D]
        key = k.transpose(1, 2)  # [B, L_txt + L_img, H, D]
        value = v.transpose(1, 2)  # [B, L_txt + L_img, H, D]

        attn_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attn_mask_tensor,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # Reshape from [B, L_img, H, D] to [B, L_img, H*D]
        batch_size, seq_len, num_heads, head_dim = attn_output.shape
        attn_output = attn_output.reshape(batch_size, seq_len, num_heads * head_dim)

        # Apply output projection
        attn_output = attn.to_out[0](attn_output)
        if len(attn.to_out) > 1:
            attn_output = attn.to_out[1](attn_output)  # dropout if present

        return attn_output

class PRXAttention(nn.Module, AttentionModuleMixin):


    _default_processor_cls = PRXAttnProcessor2_0
    _available_processors = [PRXAttnProcessor2_0]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        out_bias: bool = False,
        eps: float = 1e-6,
        processor=None,
    ):
        super().__init__()

        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim

        self.img_qkv_proj = nn.Linear(query_dim, query_dim * 3, bias=bias)

        self.norm_q = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.norm_k = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)

        self.txt_kv_proj = nn.Linear(query_dim, query_dim * 2, bias=bias)
        self.norm_added_k = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, query_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(0.0))

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )

# inspired from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
class PRXEmbedND(nn.Module):


    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def rope(self, pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0

        is_mps = pos.device.type == "mps"
        is_npu = pos.device.type == "npu"
        dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        scale = torch.arange(0, dim, 2, dtype=dtype, device=pos.device) / dim
        omega = 1.0 / (theta**scale)
        out = pos.unsqueeze(-1) * omega.unsqueeze(0)
        out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
        # Native PyTorch equivalent of: Rearrange("b n d (i j) -> b n d i j", i=2, j=2)
        # out shape: (b, n, d, 4) -> reshape to (b, n, d, 2, 2)
        out = out.reshape(*out.shape[:-1], 2, 2)
        return out.float()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [self.rope(ids[:, :, i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

class MLPEmbedder(nn.Module):


    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

class Modulation(nn.Module):


    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 6 * dim, bias=True)
        nn.init.constant_(self.lin.weight, 0)
        nn.init.constant_(self.lin.bias, 0)

    def forward(
        self, vec: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(6, dim=-1)
        return tuple(out[:3]), tuple(out[3:])

class PRXBlock(nn.Module):


    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.hidden_size = hidden_size

        # Pre-attention normalization for image tokens
        self.img_pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # PRXAttention module with built-in projections and norms
        self.attention = PRXAttention(
            query_dim=hidden_size,
            heads=num_heads,
            dim_head=self.head_dim,
            bias=False,
            out_bias=False,
            eps=1e-6,
            processor=PRXAttnProcessor2_0(),
        )

        # mlp
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        self.mlp_act = nn.GELU(approximate="tanh")

        self.modulation = Modulation(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:

        Runs modulation-gated cross-attention and MLP, with residual connections.
            **kwargs:
                Additional keyword arguments for API compatibility.
