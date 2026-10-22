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

"""PyTorch decoder for RegressLM."""

import dataclasses

import torch
from torch import nn
from torch.nn import functional as F

Tensor = torch.Tensor


@dataclasses.dataclass
class KVCache:
  """Preallocated keys/values per attention module for step-by-step decoding."""

  max_len: int
  pos: int = 0  # Self-attention positions decoded so far.
  kv: dict[nn.Module, tuple[Tensor, Tensor]] = dataclasses.field(
      default_factory=dict
  )

  def append_and_get(self, module: nn.Module, k: Tensor, v: Tensor):
    """Appends (k, v) to `module`'s buffers; returns all keys/values so far."""
    if module not in self.kv:
      shape = (*k.shape[:2], self.max_len, k.shape[3])
      self.kv[module] = (k.new_zeros(shape), v.new_zeros(shape))
    k_buf, v_buf = self.kv[module]
    end = self.pos + k.shape[2]
    k_buf[:, :, self.pos : end], v_buf[:, :, self.pos : end] = k, v
    return k_buf[:, :, :end], v_buf[:, :, :end]


def _apply_rope(x: Tensor, pos: int = 0, base: float = 10000.0) -> Tensor:
  """Applies Rotary Positional Embedding (RoPE) to (B, H, T, D_h)."""
  t, d = x.shape[-2:]
  inv_freq = base ** (-torch.arange(0, d, 2, device=x.device).float() / d)
  freqs = (pos + torch.arange(t, device=x.device).float())[:, None] * inv_freq
  cos, sin = freqs.cos(), freqs.sin()
  x0, x1 = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
  out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1).flatten(-2)
  return out.type_as(x)


class _Attention(nn.Module):
  """Multi-head attention with QK-RMSNorm, RoPE, and shared memory."""

  def __init__(self, d_model: int, nhead: int = 8, is_cross: bool = False):
    super().__init__()
    self.nhead = nhead
    self.head_dim = d_model // nhead
    self.is_cross = is_cross
    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)

  def _split(self, x: Tensor) -> Tensor:
    return x.view(*x.shape[:2], self.nhead, self.head_dim).transpose(1, 2)

  def _project_kv(self, kv: Tensor, pos: int = 0) -> tuple[Tensor, Tensor]:
    """Projects keys (normalized, rotated from `pos`) and values."""
    k = F.rms_norm(self._split(self.k_proj(kv)), (self.head_dim,), eps=1e-6)
    v = self._split(self.v_proj(kv))
    return _apply_rope(k, pos).type_as(v), v  # Autocast upcasts rms_norm.

  def _cached_kv(self, kv: Tensor, cache: KVCache) -> tuple[Tensor, Tensor]:
    """Returns all keys/values so far, projecting only the new ones."""
    if not self.is_cross:
      return cache.append_and_get(self, *self._project_kv(kv, cache.pos))
    if self not in cache.kv:  # Memory is fixed: project it once.
      cache.kv[self] = self._project_kv(kv)
    return cache.kv[self]

  def forward(
      self,
      x: Tensor,
      kv: Tensor,
      mask: Tensor | None = None,
      is_causal: bool = False,
      cache: KVCache | None = None,
  ) -> Tensor:
    bs, t, d = x.shape
    b = kv.shape[0]
    if bs % b:
      raise ValueError("Target batch must be a multiple of memory batch.")
    q = self._split(self.q_proj(x).reshape(b, -1, d))
    q = F.rms_norm(q, (self.head_dim,), eps=1e-6)
    k, v = self._project_kv(kv) if cache is None else self._cached_kv(kv, cache)
    if not self.is_cross:  # Steps are unrelated to source positions.
      q = _apply_rope(q, cache.pos if cache else 0)
    # A lone query following a cached prefix attends to all of it: no mask.
    out = F.scaled_dot_product_attention(
        q, k, v, mask, is_causal=is_causal and not (cache and cache.pos)
    )
    return self.out_proj(out.transpose(1, 2).contiguous().view(bs, t, d))


class _DecoderLayer(nn.Module):
  """Pre-RMSNorm decoder layer with QK-norm, RoPE, and Squared-ReLU FFN."""

  def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.0):
    super().__init__()
    self.norm1 = nn.RMSNorm(d_model, eps=1e-6)
    self.self_attn = _Attention(d_model, nhead)
    self.norm2 = nn.RMSNorm(d_model, eps=1e-6)
    self.cross_attn = _Attention(d_model, nhead, is_cross=True)
    self.norm3 = nn.RMSNorm(d_model, eps=1e-6)
    self.linear1 = nn.Linear(d_model, 4 * d_model)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(4 * d_model, d_model)

  def forward(
      self,
      x: Tensor,
      memory: Tensor,
      pad_mask: Tensor | None = None,
      cache: KVCache | None = None,
  ) -> Tensor:
    mask = ~pad_mask[:, None, None, :] if pad_mask is not None else None
    nx = self.norm1(x)
    x = x + self.self_attn(nx, nx, is_causal=True, cache=cache)
    x = x + self.cross_attn(self.norm2(x), memory, mask=mask, cache=cache)
    ff = F.relu(self.linear1(self.norm3(x))).square()
    return x + self.linear2(self.dropout(ff))


class Decoder(nn.Module):
  """Decoder stack with RMSNorm, QK-norm, RoPE, and Squared-ReLU FFN."""

  def __init__(self, d_model: int, num_layers: int, dropout: float = 0.0):
    super().__init__()
    make_layer = lambda: _DecoderLayer(d_model, dropout=dropout)
    self.layers = nn.ModuleList([make_layer() for _ in range(num_layers)])
    self.num_layers = num_layers
    self.norm = nn.RMSNorm(d_model, eps=1e-6)

  def forward(
      self,
      tgt: Tensor,
      memory: Tensor,
      pad_mask: Tensor | None = None,
      cache: KVCache | None = None,
  ) -> Tensor:
    for layer in self.layers:
      tgt = layer(tgt, memory, pad_mask, cache)
    if cache is not None:
      cache.pos += tgt.shape[1]
    return self.norm(tgt)
