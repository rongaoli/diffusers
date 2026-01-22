from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_utils import ModuleUtilsMixin

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin

# Modified from ClipCaptionModel in https://github...
class UniDiffuserTextDecoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):


    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.bias", r"h\.\d+\.attn\.masked_bias"]

    @register_to_config
    def __init__(
        self,
        prefix_length: int,
        prefix_inner_dim: int,
        prefix_hidden_dim: Optional[int] = None,
        vocab_size: int = 50257,  # Start of GPT2 config args
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        scale_attn_by_inverse_layer_idx: bool = False,
        reorder_and_upcast_attn: bool = False,
    ):
        super().__init__()

        self.prefix_length = prefix_length

        if prefix_inner_dim != n_embd and prefix_hidden_dim is None:
            raise ValueError(
                f"`prefix_hidden_dim` cannot be `None` when `prefix_inner_dim`: {prefix_hidden_dim} and"
                f" `n_embd`: {n_embd} are not equal."
            )

        self.prefix_inner_dim = prefix_inner_dim
        self.prefix_hidden_dim = prefix_hidden_dim

        self.encode_prefix = (
            nn.Linear(self.prefix_inner_dim, self.prefix_hidden_dim)
            if self.prefix_hidden_dim is not None
            else nn.Identity()
        )
        self.decode_prefix = (
            nn.Linear(self.prefix_hidden_dim, n_embd) if self.prefix_hidden_dim is not None else nn.Identity()
        )

        gpt_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )
        self.transformer = GPT2LMHeadModel(gpt_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        prefix_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):

        embedding_text = self.transformer.transformer.wte(input_ids)
        hidden = self.encode_prefix(prefix_embeds)
        prefix_embeds = self.decode_prefix(hidden)
        embedding_cat = torch.cat((prefix_embeds, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(input_ids.shape[0], input_ids.device)
            labels = torch.cat((dummy_token, input_ids), dim=1)
        out = self.transformer(inputs_embeds=embedding_cat, labels=labels, attention_mask=attention_mask)
        if self.prefix_hidden_dim is not None:
            return out, hidden
        else:
            return out

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def encode(self, prefix):
        return self.encode_prefix(prefix)

    @torch.no_grad()
    def generate_captions(self, features, eos_token_id, device):


        features = torch.split(features, 1, dim=0)
        generated_tokens = []
        generated_seq_lengths = []
        for feature in features:
            feature = self.decode_prefix(feature.to(device))  # back to the clip feature
            # Only support beam search for now
            output_tokens, seq_lengths = self.generate_beam(
                input_embeds=feature, device=device, eos_token_id=eos_token_id
            )
            generated_tokens.append(output_tokens[0])
            generated_seq_lengths.append(seq_lengths[0])
        generated_tokens = torch.stack(generated_tokens)
        generated_seq_lengths = torch.stack(generated_seq_lengths)
        return generated_tokens, generated_seq_lengths

    @torch.no_grad()
    def generate_beam(
        self,
        input_ids=None,
        input_embeds=None,
        device=None,
        beam_size: int = 5,
        entry_length: int = 67,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
    ):
        Generates text using the given tokenizer and text prompt or token embedding via beam search. This
        implementation is based on the beam search implementation from the [original UniDiffuser
        code](https://github.com/thu-ml/unidiffuser/blob/main/libs/caption_decoder.py#L89).
                `input_ids` and `input_embeds` must be supplied.
            device:
                The device to perform beam search on.
            beam_size (`int`, *optional*, defaults to `5`):
                The number of best states to store during beam search.
            entry_length (`int`, *optional*, defaults to `67`):
                The number of iterations to run beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The temperature to use when performing the softmax over logits from the decoding model.
