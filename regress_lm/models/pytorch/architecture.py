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

"""Default PyTorch architecture for a RegressLM."""

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


class RotaryPositionalEmbedding(nn.Module):
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


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Applies RoPE to query and key tensors."""
  # q, k have shape (B, H, L, D_h)
  # cos, sin have shape (B, H, L, D_h / 2) after broadcasting

  # Reshape q and k to separate even and odd dimensions
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


class RopeTransformerEncoderLayer(nn.Module):
  """A Transformer Encoder Layer with RoPE support."""

  def __init__(
      self, d_model: int, nhead: int, dim_feedforward: int, dropout: float
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
      rotary_pos_emb: RotaryPositionalEmbedding,
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

    # Apply RoPE to q and k
    cos, sin = rotary_pos_emb(q)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # Prepare attention mask
    final_attn_mask = (
        key_padding_mask.view(B, 1, 1, L)
        if key_padding_mask is not None
        else None
    )

    # Perform attention
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
      rotary_pos_emb: RotaryPositionalEmbedding,
      src_key_padding_mask: torch.Tensor | None,
  ) -> torch.Tensor:
    """Applies the encoder layer with the Pre-Norm structure."""
    x = src
    x = x + self.dropout1(
        self._sa_block(self.norm1(x), rotary_pos_emb, src_key_padding_mask)
    )
    x = x + self.dropout2(self._ff_block(self.norm2(x)))
    return x


class RopeEncoder(nn.Module):
  """A stack of RoPE-enabled Transformer Encoder Layers."""

  def __init__(
      self,
      encoder_layer: RopeTransformerEncoderLayer,
      num_layers: int,
      norm: nn.LayerNorm | None,
  ):
    super().__init__()
    self.layers = nn.ModuleList(
        [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
    )
    self.num_layers = num_layers
    self.norm = norm

  def forward(self, src, rotary_pos_emb, src_key_padding_mask):
    output = src
    for mod in self.layers:
      output = mod(
          output, rotary_pos_emb, src_key_padding_mask=src_key_padding_mask
      )
    if self.norm is not None:
      output = self.norm(output)
    return output


class PositionalEncoding(nn.Module):
  """Default positional encoding."""

  def __init__(self, d_model: int, max_len: int, dropout: float):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    self.register_buffer("pe", pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.dropout(x + self.pe[:, : x.size(1)])


SPD_BACKENDS = [
    nn.attention.SDPBackend.FLASH_ATTENTION,
    nn.attention.SDPBackend.MATH,
    nn.attention.SDPBackend.EFFICIENT_ATTENTION,
]


class EncoderDecoder(nn.Module):
  """Encoder-Decoder model in PyTorch."""

  def __init__(
      self,
      encoder_vocab_size: int,
      decoder_vocab_size: int,
      encoder_pad_idx: int,
      max_encoder_len: int,
      max_decoder_len: int,
      d_model: int,
      nhead: int,
      num_encoder_layers: int,
      num_decoder_layers: int,
      dim_feedforward: int,
      dropout: float,
  ):
    super().__init__()
    self.d_model = d_model
    self.encoder_pad_idx = encoder_pad_idx
    self.src_tok_emb = nn.Embedding(encoder_vocab_size, d_model)
    self.tgt_tok_emb = nn.Embedding(decoder_vocab_size, d_model)
    self.emb_dropout = nn.Dropout(dropout)

    # RoPE Encoder for best length generalization.
    self.rotary_pos_emb = RotaryPositionalEmbedding(
        d_model // nhead, max_len=max_encoder_len
    )
    encoder_layer = RopeTransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout
    )
    encoder_norm = nn.LayerNorm(d_model)
    self.encoder = RopeEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    # We use a standard positional encoding and decoder.
    self.decoder_positional_encoding = PositionalEncoding(
        d_model,
        max_len=max_decoder_len,
        dropout=dropout,
    )
    decoder_layer = nn.TransformerDecoderLayer(
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        batch_first=True,
        norm_first=True,
    )
    self.decoder = nn.TransformerDecoder(
        decoder_layer, num_layers=num_decoder_layers
    )

    self.generator = nn.Linear(d_model, decoder_vocab_size)

  def _generate_causal_mask(self, sz: int) -> torch.Tensor:
    return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

  def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
    src_padding_mask = src == self.encoder_pad_idx
    tgt_causal_mask = self._generate_causal_mask(tgt_input.size(1)).to(
        src.device
    )

    with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
      # Encode with RoPE encoder
      memory = self.encoder(
          self.emb_dropout(self.src_tok_emb(src)),
          self.rotary_pos_emb,
          src_key_padding_mask=src_padding_mask,
      )

      # Decode with standard decoder
      decoder_output = self.decoder(
          tgt=self.decoder_positional_encoding(self.tgt_tok_emb(tgt_input)),
          memory=memory,
          tgt_mask=tgt_causal_mask,
          memory_key_padding_mask=src_padding_mask,
      )
    return self.generator(decoder_output)

  def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes the source sequence using the RoPE encoder."""
    src_padding_mask = src == self.encoder_pad_idx
    src_emb = self.emb_dropout(self.src_tok_emb(src))
    with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
      memory = self.encoder(
          src_emb, self.rotary_pos_emb, src_key_padding_mask=src_padding_mask
      )
    return memory, src_padding_mask

  def next_token_logits(
      self,
      current_tgt_seq: torch.Tensor,
      memory: torch.Tensor,
      memory_key_padding_mask: torch.Tensor,
  ) -> torch.Tensor:
    """Decodes one step using the standard decoder."""
    tgt_causal_mask = self._generate_causal_mask(current_tgt_seq.size(1)).to(
        current_tgt_seq.device
    )
    tgt = self.decoder_positional_encoding(self.tgt_tok_emb(current_tgt_seq))

    with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
      decoder_output_all_steps = self.decoder(
          tgt=tgt,
          memory=memory,
          tgt_mask=tgt_causal_mask,
          memory_key_padding_mask=memory_key_padding_mask,
      )
    return self.generator(decoder_output_all_steps[:, -1, :])
