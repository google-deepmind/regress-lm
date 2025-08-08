# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Different encoder types."""

# pylint:disable=g-import-not-at-top
import abc
import copy
from typing import Literal
import mamba_ssm

import torch
from torch import nn
from torch.nn import functional as F


class BaseEncoder(nn.Module, abc.ABC):
  """Abstract base class for all encoder models."""

  @abc.abstractmethod
  def forward(
      self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None
  ) -> torch.Tensor:
    """Encodes the source sequence.

    Args:
        src: The input tensor (already embedded).
        src_key_padding_mask: The padding mask for the source tensor.

    Returns:
        The encoded memory tensor.
    """
    raise NotImplementedError("Subclasses must implement the 'forward' method.")


class _RotaryPositionalEmbedding(nn.Module):
  """Rotary Positional Embedding (RoPE)."""

  def __init__(self, d_model: int, max_len: int, base: int = 10000):
    super().__init__()
    # Note: d_model is the head dimension in our case.
    inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
    self.register_buffer("inv_freq", inv_freq)

    t = torch.arange(max_len, dtype=self.inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    # freqs shape is (max_len, d_model / 2)
    # The cached sin/cos tables should have a feature dimension of d_model / 2
    self.register_buffer(
        "cos_cached", freqs.cos()[None, None, :, :], persistent=False
    )
    self.register_buffer(
        "sin_cached", freqs.sin()[None, None, :, :], persistent=False
    )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = x.shape[2]  # x has shape (B, n_heads, L, head_dim).
    # Return tensors of shape (1, 1, L, head_dim / 2)
    return (
        self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
    )


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Applies RoPE to query and key tensors."""
  # q, k have shape (B, H, L, D_h)
  # cos, sin have shape (B, H, L, D_h / 2) after broadcasting

  # Reshape q and k to separate even and odd dimensions
  # TODO: Consider using torch.repeat_interleave.
  q_reshaped = q.float().reshape(*q.shape[:-1], -1, 2)
  k_reshaped = k.float().reshape(*k.shape[:-1], -1, 2)
  q_even, q_odd = q_reshaped[..., 0], q_reshaped[..., 1]
  k_even, k_odd = k_reshaped[..., 0], k_reshaped[..., 1]
  # q_even, q_odd, k_even, k_odd have shape (B, H, L, D_h / 2)

  # Apply rotation. All tensors in this operation have a final dim of D_h / 2.
  q_out = torch.stack(
      [q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], -1
  ).flatten(-2)
  k_out = torch.stack(
      [k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], -1
  ).flatten(-2)

  return q_out.type_as(q), k_out.type_as(k)


class _VanillaTransformerEncoderLayer(nn.Module):
  """A Transformer Encoder Layer with RoPE support."""

  def __init__(
      self,
      d_model: int,
      dim_feedforward: int,
      nhead: int,
      dropout: float,
  ):
    super().__init__()
    self.d_model = d_model
    self.nhead = nhead
    self.head_dim = d_model // nhead
    if self.head_dim * nhead != self.d_model:
      raise ValueError(
          f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
      )
    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)

    # LayerNorms
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

    # Dropouts
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.attn_dropout_p = dropout  # For F.scaled_dot_product_attention

    # Feed-forward network
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.activation = nn.ReLU()
    self.ff_dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

  def _sa_block(
      self,
      x: torch.Tensor,
      rotary_pos_emb: _RotaryPositionalEmbedding,
      key_padding_mask: torch.Tensor | None,
  ) -> torch.Tensor:
    """Self-attention block with RoPE, manual projection, and correct masking."""
    B, L, _ = x.shape  # pylint: disable=invalid-name

    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # Reshape for multi-head attention: (B, L, D) -> (B, H, L, D_h)
    q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
    k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
    v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    cos, sin = rotary_pos_emb(q)
    q, k = _apply_rotary_pos_emb(q, k, cos, sin)

    # Prepare attention mask
    final_attn_mask = (
        key_padding_mask.view(B, 1, 1, L)
        if key_padding_mask is not None
        else None
    )

    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=final_attn_mask,
        dropout_p=self.attn_dropout_p if self.training else 0.0,
    )

    # Reshape and apply output projection
    attn_output = (
        attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
    )
    return self.out_proj(attn_output)

  def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear2(self.ff_dropout(self.activation(self.linear1(x))))

  def forward(
      self,
      src: torch.Tensor,
      rotary_pos_emb: _RotaryPositionalEmbedding,
      src_key_padding_mask: torch.Tensor | None,
  ) -> torch.Tensor:
    """Applies the encoder layer with the Pre-Norm structure."""
    x = src
    x = x + self.dropout1(
        self._sa_block(self.norm1(x), rotary_pos_emb, src_key_padding_mask)
    )
    x = x + self.dropout2(self._ff_block(self.norm2(x)))
    return x


class VanillaEncoder(BaseEncoder):
  """A stack of vanilla Transformer Encoder Layers."""

  def __init__(
      self,
      d_model: int,
      num_layers: int,
      max_len: int,
      expand: int = 4,
      nhead: int = 8,
      dropout: float = 0.0,
  ):
    super().__init__()
    self.rotary_pos_emb = _RotaryPositionalEmbedding(
        d_model // nhead, max_len=max_len
    )

    encoder_layer = _VanillaTransformerEncoderLayer(
        d_model, expand * d_model, nhead, dropout
    )
    encoder_norm = nn.LayerNorm(d_model)

    self.layers = nn.ModuleList(
        [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
    )
    self.num_layers = num_layers
    self.norm = encoder_norm

  def forward(
      self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None
  ) -> torch.Tensor:
    """Forward pass conforms to the BaseEncoder interface."""
    output = src
    for mod in self.layers:
      output = mod(
          output, self.rotary_pos_emb, src_key_padding_mask=src_key_padding_mask
      )
    if self.norm is not None:
      output = self.norm(output)
    return output


class MambaEncoder(BaseEncoder):
  """Encoder built from a stack of Mamba blocks."""

  def __init__(self, d_model: int, num_layers: int, **mamba_kwargs):
    super().__init__()

    self.layers = nn.ModuleList([
        nn.Sequential(
            nn.LayerNorm(d_model),
            mamba_ssm.Mamba(d_model=d_model, **mamba_kwargs),
        )
        for _ in range(num_layers)
    ])
    # Add a final norm, which is common practice
    self.norm = nn.LayerNorm(d_model)

  def forward(
      self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None
  ) -> torch.Tensor:
    output = src
    for layer in self.layers:
      output = layer(output) + output  # Residual connection

    # Manually apply padding mask
    if src_key_padding_mask is not None:
      output = output.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

    return self.norm(output)


def get_encoder(
    encoder_type: Literal["vanilla", "mamba"],
    d_model: int,
    num_layers: int,
    max_encoder_len: int,
    **encoder_kwargs,
) -> BaseEncoder:
  """Factory function to build the specified encoder."""

  if encoder_type == "vanilla":
    return VanillaEncoder(
        d_model=d_model,
        num_layers=num_layers,
        max_len=max_encoder_len,
        **encoder_kwargs,
    )
  elif encoder_type == "mamba":
    return MambaEncoder(
        d_model=d_model,
        num_layers=num_layers,
        **encoder_kwargs,
    )

  else:
    raise ValueError(f"Unknown encoder_type: {encoder_type}")
