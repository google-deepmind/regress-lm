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

"""Hybrid exact/compressed few-shot context for encoder-decoder regression.

This module implements HybridEncoderDecoder, an encoder-decoder whose decoder
layers attend to few-shot (x, y) context through a two-tier memory:

  Tier 1 (exact): the most recent ``max_exact_pairs`` context pairs are kept
    as a token-level memory bank. Each pair contributes its full encoded x
    token sequence plus its positioned y token embeddings — no pooling, no
    loss of sequence structure. The decoder reads this tier with standard
    softmax cross-attention, so retrieval fidelity is that of full attention.

  Tier 2 (overflow): pairs evicted from the exact bank are folded into a
    fixed-size latent memory state per decoder layer via learned Perceiver
    blocks with gated forgetting (alpha/beta decay and write gates).
    The decoder queries attend to this latent array via cross-attention,
    combined with the exact bank readout via a learned per-head gate.

Each decoder layer has four pre-norm sub-layers:

  1. Causal self-attention (softmax).
  2. Softmax cross-attention to the query encoder memory (full L x d).
  3. Hybrid context attention: softmax over the exact bank, cross-attention
     over the latent memory state S, combined with a learned per-head gate.
  4. Feed-forward network.

The ``max_exact_pairs`` dial trades memory for fidelity: set it to cover the
expected product workload and the model behaves as pure uncompressed
in-context regression; overflow degrades gracefully instead of failing.

Design invariants:
  * The query x is never pooled (sub-layer 2 sees the full encoder memory).
  * Context pairs in the exact bank carry no cross-pair positional encoding,
    so bank attention is permutation-invariant over pairs.
  * Context y tokens receive positional encoding before entering either tier,
    so digit order within a numeric y is preserved (unlike a bag of tokens).
"""

import dataclasses
import enum
import math
from typing import Any

from regress_lm.pytorch import encoders
import torch
from torch import nn
from torch.nn import functional as F

# Backends attempted in order — mirrors architecture.py.
SPD_BACKENDS = [
    nn.attention.SDPBackend.FLASH_ATTENTION,
    nn.attention.SDPBackend.CUDNN_ATTENTION,
    nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    nn.attention.SDPBackend.MATH,  # Last resort, materializes whole matrix.
]

# pylint: disable=invalid-name,g-missing-property-docstring,g-doc-args,protected-access


class OverflowMode(enum.Enum):
  """Compression strategy for pairs evicted from the exact bank.

  Attributes:
    LATENT_ARRAY: Perceiver-style learned latent bottleneck. A fixed array of N
      latent vectors cross-attends to evicted tokens with gated decay and write
      updates. Nonlinear, learned compression.
  """

  LATENT_ARRAY = "latent_array"


class _PositionalEncoding(nn.Module):
  """Sinusoidal positional encoding."""

  def __init__(self, d_model: int, max_len: int):
    super().__init__()
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    self.register_buffer("pe", pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Context state container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HybridContextState:
  """Two-tier context memory.

  Attributes:
    bank: (B, M, d_model) token-level memory of exact pairs, or None when every
      pair has been compressed (or no pairs exist). M is the total token count
      over all exact pairs.
    bank_mask: (B, M) bool, True marks padding tokens inside the bank.
    pair_slices: Column ranges [(start, end), ...] of each stored pair inside
      the bank, oldest first. Shared across the batch (pairs are padded to a
      uniform width).
    S: Per-decoder-layer latent memory states, each (B, num_latents, d_model),
      or None if nothing has been compressed yet.
    num_pairs_total: Total pairs absorbed (exact + compressed).
  """

  bank: torch.Tensor | None
  bank_mask: torch.Tensor | None
  pair_slices: list[tuple[int, int]]
  S: list[torch.Tensor] | None
  num_pairs_total: int

  @property
  def num_exact_pairs(self) -> int:
    return len(self.pair_slices)

  def detached(self) -> "HybridContextState":
    """Returns a gradient-detached copy (for inference-time reuse)."""
    d = lambda t: None if t is None else t.detach()
    dl = lambda ts: None if ts is None else [t.detach() for t in ts]
    return HybridContextState(
        bank=d(self.bank),
        bank_mask=d(self.bank_mask),
        pair_slices=list(self.pair_slices),
        S=dl(self.S),
        num_pairs_total=self.num_pairs_total,
    )

  def expand_for_queries(self, num_queries: int) -> "HybridContextState":
    """Repeats the batch dim so one state serves Q queries per episode."""
    r = lambda t: (
        None if t is None else t.repeat_interleave(num_queries, dim=0)
    )
    rl = lambda ts: (
        None
        if ts is None
        else [t.repeat_interleave(num_queries, dim=0) for t in ts]
    )
    return HybridContextState(
        bank=r(self.bank),
        bank_mask=r(self.bank_mask),
        pair_slices=list(self.pair_slices),
        S=rl(self.S),
        num_pairs_total=self.num_pairs_total,
    )


# ---------------------------------------------------------------------------
# Hybrid context attention sub-layer
# ---------------------------------------------------------------------------


class _HybridContextAttention(nn.Module):
  """Softmax attention over exact bank + Perceiver latent memory readout.

  The two readouts are combined with a learned per-head sigmoid gate.
  The gate bias is initialized negative so early training relies on the exact
  bank; the compressed latent memory pathway fades in as the write/read
  attentions become useful.
  """

  def __init__(
      self,
      d_model: int,
      nhead: int,
      num_latents: int = 32,
      num_write_layers: int = 4,
  ):
    """Initializes hybrid context attention.

    Args:
      d_model: Model hidden dimension.
      nhead: Number of attention heads.
      num_latents: Number of latent memory vectors.
      num_write_layers: Number of Perceiver cross/self attention layers.
    """
    super().__init__()
    self.d_model = d_model
    self.nhead = nhead
    self.head_dim = d_model // nhead
    self.num_latents = num_latents
    self.num_write_layers = num_write_layers

    # Shared K/V projections for exact bank attention.
    self.W_k = nn.Linear(d_model, d_model, bias=False)
    self.W_v = nn.Linear(d_model, d_model, bias=False)
    self.W_q = nn.Linear(d_model, d_model, bias=False)
    self.W_out = nn.Linear(d_model, d_model, bias=False)

    # Per-head gate between bank readout and compressed readout.
    # sigmoid(-2) ~= 0.12: start mostly on the exact bank.
    self.gate = nn.Parameter(torch.full((nhead,), -2.0))

    # Perceiver latent memory.
    self.latent_init = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
    self.latent_write_attns = nn.ModuleList([
        nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        for _ in range(self.num_write_layers)
    ])
    self.latent_write_norms = nn.ModuleList(
        [nn.LayerNorm(d_model) for _ in range(self.num_write_layers)]
    )
    self.latent_self_attns = nn.ModuleList([
        nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        for _ in range(self.num_write_layers)
    ])
    self.latent_self_norms = nn.ModuleList(
        [nn.LayerNorm(d_model) for _ in range(self.num_write_layers)]
    )

    # Gated DeltaNet-style: separate decay (alpha) and write (beta) gates.
    # alpha controls global forgetting, beta controls update strength.
    # Init: alpha -> 1 (retain), beta -> 0 (no write) -> identity at init.
    self.decay_gate_proj = nn.Linear(d_model, d_model)
    nn.init.constant_(self.decay_gate_proj.bias, 5.0)  # sigmoid(5) ~= 0.993
    self.write_gate_proj = nn.Linear(d_model, d_model)
    nn.init.constant_(self.write_gate_proj.bias, -5.0)  # sigmoid(-5) ~= 0.007

    # Post-compression LayerNorm: stabilizes latent state magnitude
    # across many sequential compressions (prevents drift / oscillation).
    self.latent_state_norm = nn.LayerNorm(d_model)

    # FFN after cross-attention.
    self.latent_cross_ffs = nn.ModuleList([
        nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        for _ in range(self.num_write_layers)
    ])
    # FFN after self-attention.
    self.latent_self_ffs = nn.ModuleList([
        nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        for _ in range(self.num_write_layers)
    ])

    # Read: decoder queries attend to latent array.
    self.latent_read_attn = nn.MultiheadAttention(
        d_model, nhead, batch_first=True, dropout=0.0
    )

    self._layer_state: torch.Tensor | None = None

  def _compress_latent(
      self,
      pair_tokens: torch.Tensor,
      pair_mask: torch.Tensor | None,
      S_prev: torch.Tensor | None,
  ) -> torch.Tensor:
    """Perceiver: latent array cross-attends to evicted pair tokens.

    On first call, initializes from learned ``latent_init``. Each eviction
    runs ``num_write_layers`` blocks of (cross-attention + FFN + self-attention
    + FFN) with residual connections, followed by alpha/beta gating and
    post-compression layer normalization.

    Args:
      pair_tokens: (B, L, d_model) token block of the evicted pair.
      pair_mask: (B, L) bool, True = padding.
      S_prev: Prior latent state (B, num_latents, d_model), or None.

    Returns:
      Updated latent state S of shape (B, num_latents, d_model).
    """
    B = pair_tokens.shape[0]
    if S_prev is None:
      S_prev = self.latent_init.unsqueeze(0).expand(B, -1, -1)

    latents = S_prev
    key_padding_mask = pair_mask  # True = ignore for MHA.

    for i in range(self.num_write_layers):
      # 1. Cross-attention: latents query the evicted pair tokens.
      normed_latents = self.latent_write_norms[i](latents)
      attn_out, _ = self.latent_write_attns[i](
          normed_latents,
          pair_tokens,
          pair_tokens,
          key_padding_mask=key_padding_mask,
      )
      latents = latents + attn_out

      # 2. FFN after cross-attention.
      latents = latents + self.latent_cross_ffs[i](latents)

      # 3. Perceiver self-attention: latents communicate with each other.
      normed_latents_self = self.latent_self_norms[i](latents)
      self_attn_out, _ = self.latent_self_attns[i](
          normed_latents_self, normed_latents_self, normed_latents_self
      )
      latents = latents + self_attn_out

      # 4. FFN after self-attention.
      latents = latents + self.latent_self_ffs[i](latents)

    # Gated DeltaNet-style update with separate decay (alpha) and write (beta).
    # S_new = alpha * S_prev + beta * (latents - alpha * S_prev)
    alpha = torch.sigmoid(self.decay_gate_proj(latents))  # decay in (0, 1)
    beta = torch.sigmoid(self.write_gate_proj(latents))  # write in (0, 1)
    decayed = alpha * S_prev
    delta = latents - decayed
    final_latents = self.latent_state_norm(decayed + beta * delta)

    return final_latents

  def compress_pair(
      self,
      pair_tokens: torch.Tensor,
      pair_mask: torch.Tensor | None,
      S_prev: torch.Tensor | None,
  ) -> torch.Tensor:
    """Folds one evicted pair's token block into the latent memory state.

    Args:
      pair_tokens: (B, L, d_model) token block of the evicted pair.
      pair_mask: (B, L) bool, True = padding.
      S_prev: Prior latent state (B, num_latents, d_model), or None.

    Returns:
      Updated latent state S of shape (B, num_latents, d_model).
    """
    use_ckpt = torch.is_grad_enabled() and (
        pair_tokens.requires_grad
        or (S_prev is not None and S_prev.requires_grad)
    )
    if use_ckpt:
      return torch.utils.checkpoint.checkpoint(
          self._compress_latent,
          pair_tokens,
          pair_mask,
          S_prev,
          use_reentrant=False,
      )
    return self._compress_latent(pair_tokens, pair_mask, S_prev)

  def forward(
      self,
      decoder_hidden: torch.Tensor,
      state: HybridContextState,
  ) -> torch.Tensor:
    """Reads both tiers and gate-combines them.

    Args:
      decoder_hidden: (B, L_dec, d_model) current decoder hidden states.
      state: The hybrid context state (at least one tier must be non-empty).

    Returns:
      (B, L_dec, d_model) context-attended output.
    """
    B, L_dec, _ = decoder_hidden.shape
    H, dh = self.nhead, self.head_dim
    q = self.W_q(decoder_hidden).view(B, L_dec, H, dh)

    # Tier 1: exact bank readout (softmax cross-attention).
    bank_out = None
    if state.bank is not None and state.bank.shape[1] > 0:
      k = self.W_k(state.bank).view(B, -1, H, dh)
      v = self.W_v(state.bank).view(B, -1, H, dh)
      qh = q.transpose(1, 2)
      kh = k.transpose(1, 2)
      vh = v.transpose(1, 2)
      attn_mask = None
      if state.bank_mask is not None:
        attn_mask = ~state.bank_mask[:, None, None, :]
      with nn.attention.sdpa_kernel(SPD_BACKENDS):
        bank_out = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=attn_mask
        ).transpose(
            1, 2
        )  # (B, L_dec, H, dh)

    # Tier 2: latent memory readout.
    comp_out = None
    if state.S is not None:
      S = self._layer_state
      # Perceiver readout: decoder queries attend to latent array.
      # S is (B, num_latents, d_model); operate in d_model space.
      latent_out, _ = self.latent_read_attn(decoder_hidden, S, S)
      # Reshape to (B, L_dec, H, dh) for gate combination.
      comp_out = latent_out.view(B, L_dec, H, dh)

    # Combine tiers with learned gate.
    if bank_out is None and comp_out is None:
      raise ValueError("Hybrid attention called with an empty state.")
    if comp_out is None:
      out = bank_out
    elif bank_out is None:
      out = comp_out
    else:
      g = torch.sigmoid(self.gate).view(1, 1, H, 1)
      out = (1.0 - g) * bank_out + g * comp_out

    assert out is not None  # Guaranteed by ValueError guard above.
    return self.W_out(out.reshape(B, L_dec, self.d_model))

  def set_layer_state(self, S: torch.Tensor | None) -> None:
    """Stashes this layer's latent state S for the next forward call."""
    self._layer_state = S


# ---------------------------------------------------------------------------
# Decoder layer (4 sub-layers)
# ---------------------------------------------------------------------------


class _HybridDecoderLayer(nn.Module):
  """Pre-norm decoder layer: self-attn, query cross-attn, context, FFN."""

  def __init__(
      self,
      d_model: int,
      nhead: int,
      dim_feedforward: int,
      overflow_mode: OverflowMode | str = OverflowMode.LATENT_ARRAY,
      dropout: float = 0.0,
      num_latents: int = 32,
      num_write_layers: int = 4,
      **unused_kwargs: Any,
  ):
    super().__init__()
    del overflow_mode, unused_kwargs
    self.self_attn = nn.MultiheadAttention(
        d_model, nhead, batch_first=True, dropout=0.0
    )
    self.query_cross_attn = nn.MultiheadAttention(
        d_model, nhead, batch_first=True, dropout=0.0
    )
    self.context_attn = _HybridContextAttention(
        d_model,
        nhead,
        num_latents=num_latents,
        num_write_layers=num_write_layers,
    )
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.linear2 = nn.Linear(dim_feedforward, d_model)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.norm4 = nn.LayerNorm(d_model)
    self.ffn_dropout = nn.Dropout(dropout)

  def forward(
      self,
      tgt: torch.Tensor,
      query_memory: torch.Tensor,
      context_state: HybridContextState | None,
      tgt_mask: torch.Tensor | None = None,
      tgt_is_causal: bool = False,
      memory_key_padding_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    x = tgt

    # 1. Causal self-attention.
    x2 = self.norm1(x)
    x = (
        x
        + self.self_attn(
            x2, x2, x2, attn_mask=tgt_mask, is_causal=tgt_is_causal
        )[0]
    )

    # 2. Softmax cross-attention to the full query encoder memory.
    x2 = self.norm2(x)
    x = (
        x
        + self.query_cross_attn(
            x2,
            query_memory,
            query_memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
    )

    # 3. Hybrid context attention (skipped at k=0).
    if context_state is not None and (
        context_state.bank is not None or context_state.S is not None
    ):
      x2 = self.norm3(x)
      x = x + self.context_attn(x2, context_state)

    # 4. FFN.
    x2 = self.norm4(x)
    x = x + self.linear2(self.ffn_dropout(F.relu(self.linear1(x2))))
    return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class HybridEncoderDecoder(nn.Module):
  """Encoder-decoder with a two-tier (exact + compressed) few-shot memory.

  ``max_exact_pairs`` is the few-shot dial: up to that many pairs are held
  with full token-level structure and read via softmax attention; older
  pairs are folded into a fixed-size per-layer latent memory state.

  Usage::

      model = HybridEncoderDecoder(..., max_exact_pairs=64)
      state = model.build_context_state(ctx_src, ctx_y)   # (B, k, L), (B, k, T)
      state = model.update_context_state(state, new_src, new_y)
      logits = model(query_src, tgt_input, context_state=state)
  """

  def __init__(
      self,
      encoder_vocab_size: int,
      decoder_vocab_size: int,
      encoder_pad_idx: int,
      max_encoder_len: int,
      max_decoder_len: int,
      d_model: int,
      num_encoder_layers: int,
      num_decoder_layers: int,
      max_exact_pairs: int = 64,
      overflow_mode: OverflowMode | str = OverflowMode.LATENT_ARRAY,
      decoder_dropout: float = 0.0,
      encoder_type: encoders.EncoderType = encoders.EncoderType.VANILLA,
      additional_encoder_kwargs: dict[str, Any] | None = None,
      nhead: int = 8,
      num_latents: int = 32,
      num_write_layers: int = 4,
      **unused_kwargs: Any,
  ):
    """Initializes the HybridEncoderDecoder.

    Args:
      encoder_vocab_size: Vocabulary size for the encoder.
      decoder_vocab_size: Vocabulary size for the decoder.
      encoder_pad_idx: Padding token index used by the encoder.
      max_encoder_len: Maximum source sequence length.
      max_decoder_len: Maximum target sequence length.
      d_model: Model hidden dimension.
      num_encoder_layers: Number of encoder layers.
      num_decoder_layers: Number of decoder layers.
      max_exact_pairs: Exact-tier capacity in pairs. The dial.
      overflow_mode: Overflow compression strategy (see ``OverflowMode``).
      decoder_dropout: FFN dropout inside decoder layers.
      encoder_type: Encoder architecture.
      additional_encoder_kwargs: Extra kwargs for the encoder factory.
      nhead: Attention heads in decoder layers.
      num_latents: Number of latent memory vectors.
      num_write_layers: Number of Perceiver cross/self attention layers.
      **unused_kwargs: Additional kwargs (ignored for backwards compatibility).
    """
    super().__init__()
    del unused_kwargs
    if max_exact_pairs < 0:
      raise ValueError("max_exact_pairs must be >= 0.")
    self.encoder_pad_idx = encoder_pad_idx
    self.max_exact_pairs = max_exact_pairs

    if isinstance(overflow_mode, str):
      overflow_mode = OverflowMode(overflow_mode)
    if overflow_mode != OverflowMode.LATENT_ARRAY:
      raise ValueError(
          f"Only OverflowMode.LATENT_ARRAY is supported, got {overflow_mode}."
      )
    self.overflow_mode = overflow_mode

    self.encoder = encoder_type.make(
        vocab_size=encoder_vocab_size,
        d_model=d_model,
        num_layers=num_encoder_layers,
        max_encoder_len=max_encoder_len,
        **(additional_encoder_kwargs or {}),
    )
    hidden_dim = self.encoder.hidden_dim

    self.tgt_tok_emb = nn.Embedding(decoder_vocab_size, hidden_dim)
    self.decoder_positional_encoding = _PositionalEncoding(
        hidden_dim, max_len=max_decoder_len
    )
    # Type embeddings distinguishing context-x vs context-y tokens in the
    # bank.
    self.ctx_type_emb = nn.Embedding(2, hidden_dim)
    nn.init.normal_(self.ctx_type_emb.weight, std=0.02)
    # Pair positional encoding: tokens from pair i share a sinusoidal vector,
    # binding xᵢ to yᵢ. Fixed (not learned) so it generalizes to any memory
    # size without retraining.
    max_pair_len = max(max_exact_pairs + 1, 256)
    pair_pe = torch.zeros(max_pair_len, hidden_dim)
    pos = torch.arange(max_pair_len).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
    )
    pair_pe[:, 0::2] = torch.sin(pos * div)
    pair_pe[:, 1::2] = torch.cos(pos * div)
    self.register_buffer("pair_pe", pair_pe)
    self.pair_pe_scale = nn.Parameter(torch.tensor(0.1))

    self.decoder_layers = nn.ModuleList([
        _HybridDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            overflow_mode=overflow_mode,
            dropout=decoder_dropout,
            num_latents=num_latents,
            num_write_layers=num_write_layers,
        )
        for _ in range(num_decoder_layers)
    ])
    self.generator = nn.Linear(hidden_dim, decoder_vocab_size)

  # ---- Encoding ------------------------------------------------------------

  def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes (B, L) token ids to ((B, L, d), (B, L) padding mask)."""
    src_padding_mask = src == self.encoder_pad_idx
    with nn.attention.sdpa_kernel(SPD_BACKENDS):
      memory = self.encoder(src, src_key_padding_mask=src_padding_mask)
    return memory, src_padding_mask

  def _make_pair_block(
      self, src: torch.Tensor, y_token_ids: torch.Tensor, pair_idx: int
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds one pair's token-level block.

    The x side is the full encoder output; the y side is embedded WITH
    positional encoding (digit order is preserved) — then both get type
    embeddings + a shared pair positional embedding and are concatenated.

    Args:
      src: (B, L_x) encoder token ids for x.
      y_token_ids: (B, T_y) decoder token ids for y.
      pair_idx: Index of this pair in the bank (used for pair binding).

    Returns:
      (tokens, mask): (B, L_x + T_y, d) block and (B, L_x + T_y) pad mask.
    """
    enc_x, x_pad = self.encode(src)
    enc_x = enc_x.to(dtype=self.tgt_tok_emb.weight.dtype)
    emb_y = self.decoder_positional_encoding(self.tgt_tok_emb(y_token_ids))

    pair_emb = self.pair_pe[pair_idx] * self.pair_pe_scale  # (d,)
    enc_x = enc_x + self.ctx_type_emb.weight[0] + pair_emb
    emb_y = emb_y + self.ctx_type_emb.weight[1] + pair_emb

    tokens = torch.cat([enc_x, emb_y], dim=1)
    y_pad = torch.zeros(emb_y.shape[:2], dtype=torch.bool, device=emb_y.device)
    mask = torch.cat([x_pad, y_pad], dim=1)
    return tokens, mask

  # ---- State construction --------------------------------------------------

  def _empty_state(self) -> HybridContextState:
    return HybridContextState(
        bank=None,
        bank_mask=None,
        pair_slices=[],
        S=None,
        num_pairs_total=0,
    )

  def _compress_into_layers(
      self,
      state: HybridContextState,
      pair_tokens: torch.Tensor,
      pair_mask: torch.Tensor,
  ) -> list[torch.Tensor]:
    """Folds one pair block into every layer's latent overflow state."""
    new_S: list[torch.Tensor] = []
    for l, layer in enumerate(self.decoder_layers):
      S_prev = state.S[l] if state.S is not None else None
      S_l = layer.context_attn.compress_pair(pair_tokens, pair_mask, S_prev)
      new_S.append(S_l)
    return new_S

  def _append_pair(
      self,
      state: HybridContextState,
      pair_tokens: torch.Tensor,
      pair_mask: torch.Tensor,
  ) -> HybridContextState:
    """Appends a pair to the bank, evicting the oldest pair on overflow."""
    if state.bank is None:
      bank, bank_mask = pair_tokens, pair_mask
      slices = [(0, pair_tokens.shape[1])]
    else:
      start = state.bank.shape[1]
      bank = torch.cat([state.bank, pair_tokens], dim=1)
      bank_mask = torch.cat([state.bank_mask, pair_mask], dim=1)
      slices = state.pair_slices + [(start, start + pair_tokens.shape[1])]

    S = state.S
    if len(slices) > self.max_exact_pairs:
      # Evict the oldest pair into tier 2 (latent memory compression).
      ev_start, ev_end = slices[0]
      evicted_tokens = bank[:, ev_start:ev_end]
      evicted_mask = bank_mask[:, ev_start:ev_end]
      S = self._compress_into_layers(state, evicted_tokens, evicted_mask)
      bank = bank[:, ev_end:]
      bank_mask = bank_mask[:, ev_end:]
      slices = [(s - ev_end, e - ev_end) for (s, e) in slices[1:]]

    return HybridContextState(
        bank=bank,
        bank_mask=bank_mask,
        pair_slices=slices,
        S=S,
        num_pairs_total=state.num_pairs_total + 1,
    )

  def build_context_state(
      self,
      context_src: torch.Tensor,
      context_y_token_ids: torch.Tensor,
  ) -> HybridContextState:
    """Builds the two-tier state from k context pairs.

    The newest ``max_exact_pairs`` pairs land in the exact bank; older pairs
    are compressed oldest-first into the per-layer latent memory states.

    Args:
      context_src: (B, k, L) encoder token ids for the k context x's.
      context_y_token_ids: (B, k, T) decoder token ids for the k context y's.

    Returns:
      A ``HybridContextState``. With k = 0, an empty state (forward() then
      skips the context sub-layer).
    """
    state = self._empty_state()
    k = context_src.shape[1]
    for i in range(k):
      tokens, mask = self._make_pair_block(
          context_src[:, i], context_y_token_ids[:, i], pair_idx=i
      )
      state = self._append_pair(state, tokens, mask)
    return state

  def update_context_state(
      self,
      state: HybridContextState,
      new_src: torch.Tensor,
      new_y_token_ids: torch.Tensor,
  ) -> HybridContextState:
    """Incrementally adds one (x, y) pair.

    Cost: one encoder pass over the new x plus a bank append; on overflow,
    additionally folds the evicted pair into the latent memory.

    Args:
      state: Existing state.
      new_src: (B, L) encoder token ids for the new x.
      new_y_token_ids: (B, T) decoder token ids for the new y.

    Returns:
      Updated ``HybridContextState``.
    """
    pair_idx = state.num_pairs_total
    tokens, mask = self._make_pair_block(new_src, new_y_token_ids, pair_idx)
    return self._append_pair(state, tokens, mask)

  # ---- Forward passes ------------------------------------------------------

  def _run_decoder(
      self,
      tgt_input: torch.Tensor,
      memory: torch.Tensor,
      memory_padding_mask: torch.Tensor,
      context_state: HybridContextState | None,
  ) -> torch.Tensor:
    tgt = self.decoder_positional_encoding(self.tgt_tok_emb(tgt_input))
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
        tgt_input.size(1), device=tgt_input.device, dtype=torch.bool
    )
    if context_state is not None and context_state.S is not None:
      for l, layer in enumerate(self.decoder_layers):
        layer.context_attn.set_layer_state(context_state.S[l])

    x = tgt
    for layer in self.decoder_layers:
      x = layer(
          x,
          memory.to(dtype=self.tgt_tok_emb.weight.dtype),
          context_state,
          tgt_mask=tgt_mask,
          tgt_is_causal=True,
          memory_key_padding_mask=memory_padding_mask,
      )
    return x

  def forward(
      self,
      src: torch.Tensor,
      tgt_input: torch.Tensor,
      context_state: HybridContextState | None = None,
  ) -> torch.Tensor:
    """Training forward pass.

    Args:
      src: (B, L_src) encoder token ids for the query x.
      tgt_input: (B, L_tgt) teacher-forced decoder input ids.
      context_state: Optional hybrid state (None or empty = k=0).

    Returns:
      (B, L_tgt, vocab) logits.
    """
    memory, memory_padding_mask = self.encode(src)
    x = self._run_decoder(tgt_input, memory, memory_padding_mask, context_state)
    return self.generator(x)

  def next_token_logits(
      self,
      current_tgt_seq: torch.Tensor,
      memory: torch.Tensor,
      memory_key_padding_mask: torch.Tensor,
      context_state: HybridContextState | None = None,
  ) -> torch.Tensor:
    """(B, vocab) logits for the next token during decoding."""
    x = self._run_decoder(
        current_tgt_seq, memory, memory_key_padding_mask, context_state
    )
    return self.generator(x[:, -1, :])
