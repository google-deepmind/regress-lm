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

import torch
from torch import nn
from torch.nn import functional as F


def _apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
  """Applies Rotary Positional Embedding (RoPE) to (B, H, T, D_h)."""
  t, d = x.shape[-2:]
  inv_freq = base ** (-torch.arange(0, d, 2, device=x.device).float() / d)
  freqs = torch.arange(t, device=x.device).float()[:, None] * inv_freq
  cos, sin = freqs.cos(), freqs.sin()
  x0, x1 = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
  out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1).flatten(-2)
  return out.type_as(x)


class _Attention(nn.Module):
  """Multi-head attention with QK-RMSNorm, optional RoPE, and shared memory."""

  def __init__(self, d_model: int, nhead: int = 8, use_rope: bool = False):
    super().__init__()
    self.nhead = nhead
    self.head_dim = d_model // nhead
    self.use_rope = use_rope
    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)

  def _split(self, x: torch.Tensor) -> torch.Tensor:
    return x.view(*x.shape[:2], self.nhead, self.head_dim).transpose(1, 2)

  def forward(
      self,
      x: torch.Tensor,
      kv: torch.Tensor,
      mask: torch.Tensor | None = None,
      is_causal: bool = False,
  ) -> torch.Tensor:
    bs, t, d = x.shape
    b = kv.shape[0]
    if bs % b:
      raise ValueError("Target batch must be a multiple of memory batch.")
    q = self._split(self.q_proj(x).reshape(b, -1, d))
    k, v = self._split(self.k_proj(kv)), self._split(self.v_proj(kv))
    q = F.rms_norm(q, (self.head_dim,), eps=1e-6)
    k = F.rms_norm(k, (self.head_dim,), eps=1e-6)
    if self.use_rope:
      q, k = _apply_rope(q), _apply_rope(k)
    out = F.scaled_dot_product_attention(q, k, v, mask, is_causal=is_causal)
    return self.out_proj(out.transpose(1, 2).contiguous().view(bs, t, d))


class _DecoderLayer(nn.Module):
  """Pre-RMSNorm decoder layer with QK-norm, RoPE, and Squared-ReLU FFN."""

  def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.0):
    super().__init__()
    self.norm1 = nn.RMSNorm(d_model, eps=1e-6)
    self.self_attn = _Attention(d_model, nhead, use_rope=True)
    self.norm2 = nn.RMSNorm(d_model, eps=1e-6)
    self.cross_attn = _Attention(d_model, nhead, use_rope=False)
    self.norm3 = nn.RMSNorm(d_model, eps=1e-6)
    self.linear1 = nn.Linear(d_model, 4 * d_model)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(4 * d_model, d_model)

  def forward(
      self,
      x: torch.Tensor,
      memory: torch.Tensor,
      pad_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    mask = ~pad_mask[:, None, None, :] if pad_mask is not None else None
    nx = self.norm1(x)
    x = x + self.self_attn(nx, nx, is_causal=True)
    x = x + self.cross_attn(self.norm2(x), memory, mask=mask)
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
      tgt: torch.Tensor,
      memory: torch.Tensor,
      pad_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    for layer in self.layers:
      tgt = layer(tgt, memory, pad_mask)
    return self.norm(tgt)
