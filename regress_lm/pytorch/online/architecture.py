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

"""Incremental encoder-decoder with linear attention context state.

This module implements IncrementalEncoderDecoder, an encoder-decoder that
supports efficient incremental (x, y) pair updates via a linear attention
context state S. Each decoder layer has four sub-layers:

  1. Causal self-attention (standard softmax).
  2. Softmax cross-attention to the query encoder memory (full L×d).
  3. Linear cross-attention to the accumulated context state S (NEW).
  4. Feed-forward network (standard).

The context state is built by encoding each context x_i through the shared
encoder, embedding each y_i through the shared decoder token embedding,
concatenating, and accumulating outer-product statistics using the ELU+1
feature map (Katharopoulos et al., 2020).
"""

import math
from typing import Any, NamedTuple

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

# pylint: disable=invalid-name


class _PositionalEncoding(nn.Module):
  """Sinusoidal positional encoding (same as architecture.py)."""

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


class ContextState(NamedTuple):
  """Accumulated linear-attention context state for all decoder layers.

  Attributes:
    S: List of (B, H, m, d_v) tensors — one outer-product accumulator per
      decoder layer.
    z: List of (B, H, m) tensors — one normalizer accumulator per decoder layer.
  """

  S: list[torch.Tensor]
  z: list[torch.Tensor]


# ---------------------------------------------------------------------------
# Linear cross-attention sub-layer
# ---------------------------------------------------------------------------


class _LinearCrossAttention(nn.Module):
  """Linear cross-attention to a pre-accumulated context state S.

  Uses the ELU+1 feature map from Katharopoulos et al. (2020) so that
  cross-attention decomposes into a state S = Σ φ(k)ᵀv and a normalizer
  z = Σ φ(k), allowing O(1) incremental updates per new (x, y) pair.
  """

  def __init__(
      self,
      d_model: int,
      nhead: int,
      num_features: int | None = None,
  ):
    """Initializes the linear cross-attention sub-layer.

    Args:
      d_model: Model hidden dimension.
      nhead: Number of attention heads.
      num_features: Feature-map output dimension per head.  Defaults to
        ``d_model`` (i.e. ``head_dim`` per head after reshape).
    """
    super().__init__()
    self.d_model = d_model
    self.nhead = nhead
    self.head_dim = d_model // nhead
    # num_features is unused with ELU+1 (identity-dimension feature map),
    # but kept for API compatibility with random-feature variants.
    self.num_features = num_features or d_model

    self.W_k_ctx = nn.Linear(d_model, d_model, bias=False)
    self.W_v_ctx = nn.Linear(d_model, d_model, bias=False)
    self.W_q_ctx = nn.Linear(d_model, d_model, bias=False)
    self.W_out_ctx = nn.Linear(d_model, d_model, bias=False)

  # ---- feature map --------------------------------------------------------

  @staticmethod
  def feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1 feature map: φ(x) = elu(x) + 1."""
    return F.elu(x, alpha=1.0) + 1.0

  # ---- state construction --------------------------------------------------

  def build_state(
      self,
      context_tokens: torch.Tensor,
      padding_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds the linear-attention state from context tokens.

    Args:
      context_tokens: (B, L, d_model) — concatenated encoder-output and embedded
        y tokens for a single context pair.
      padding_mask: (B, L) bool tensor where ``True`` marks padding positions
        that should be ignored.

    Returns:
      A tuple ``(S, z)`` where
        S: (B, H, m, d_v) outer-product accumulator.
        z: (B, H, m) normalizer accumulator.
    """
    B, L, _ = context_tokens.shape  # pylint: disable=invalid-name
    H = self.nhead  # pylint: disable=invalid-name

    k = self.W_k_ctx(context_tokens).view(B, L, H, self.head_dim)
    v = self.W_v_ctx(context_tokens).view(B, L, H, self.head_dim)

    k_prime = self.feature_map(k)  # (B, L, H, head_dim)

    # Zero out padding positions so they don't contribute to the state.
    if padding_mask is not None:
      # padding_mask: (B, L) — True = pad.  Expand to (B, L, 1, 1).
      k_prime = k_prime.masked_fill(
          padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0
      )

    # S = Σ_l φ(k_l) ⊗ v_l  →  (B, H, m, d_v)
    S = torch.einsum("blhm, blhd -> bhmd", k_prime, v)
    # z = Σ_l φ(k_l)  →  (B, H, m)
    z = torch.einsum("blhm -> bhm", k_prime)

    return S, z

  # ---- forward (query the state) ------------------------------------------

  def forward(
      self,
      decoder_hidden: torch.Tensor,
      S: torch.Tensor,
      z: torch.Tensor,
  ) -> torch.Tensor:
    """Queries the accumulated context state.

    Args:
      decoder_hidden: (B, L_dec, d_model) — current decoder hidden states.
      S: (B, H, m, d_v) — outer-product accumulator from ``build_state``.
      z: (B, H, m) — normalizer accumulator from ``build_state``.

    Returns:
      (B, L_dec, d_model) — context-attended output.
    """
    B, L_dec, _ = decoder_hidden.shape  # pylint: disable=invalid-name

    q = self.W_q_ctx(decoder_hidden).view(B, L_dec, self.nhead, self.head_dim)
    q_prime = self.feature_map(q)  # (B, L_dec, H, head_dim)

    # numerator = φ(q) · S  →  (B, L_dec, H, d_v)
    numerator = torch.einsum("blhm, bhmd -> blhd", q_prime, S)
    # denominator = φ(q) · z  →  (B, L_dec, H)
    denominator = torch.einsum("blhm, bhm -> blh", q_prime, z)

    output = numerator / (denominator.unsqueeze(-1) + 1e-6)
    output = output.reshape(B, L_dec, self.d_model)
    return self.W_out_ctx(output)


# ---------------------------------------------------------------------------
# Incremental decoder layer (4 sub-layers)
# ---------------------------------------------------------------------------


class _IncrementalDecoderLayer(nn.Module):
  """A single decoder layer with four pre-norm sub-layers.

  Sub-layers:
    1. Causal self-attention (standard softmax via ``nn.MultiheadAttention``).
    2. Softmax cross-attention to the query encoder memory.
    3. Linear cross-attention to the accumulated context state S.
    4. Position-wise feed-forward network.
  """

  def __init__(
      self,
      d_model: int,
      nhead: int,
      dim_feedforward: int,
      num_features: int | None = None,
      dropout: float = 0.0,
  ):
    """Initializes the incremental decoder layer.

    Args:
      d_model: Model hidden dimension.
      nhead: Number of attention heads.
      dim_feedforward: Inner dimension of the FFN.
      num_features: Feature-map dimension for linear cross-attention.
      dropout: FFN internal dropout rate.
    """
    super().__init__()

    # Sub-layer 1: causal self-attention.
    self.self_attn = nn.MultiheadAttention(
        d_model,
        nhead,
        batch_first=True,
        dropout=0.0,
    )
    # Sub-layer 2: softmax cross-attention to query memory.
    self.query_cross_attn = nn.MultiheadAttention(
        d_model,
        nhead,
        batch_first=True,
        dropout=0.0,
    )
    # Sub-layer 3: linear cross-attention to context state.
    self.context_cross_attn = _LinearCrossAttention(
        d_model, nhead, num_features
    )
    # Sub-layer 4: position-wise FFN.
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    # Pre-norm layer norms (one per sub-layer).
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.norm4 = nn.LayerNorm(d_model)

    # FFN internal dropout only; all residual / attention dropouts are 0.
    self.ffn_dropout = nn.Dropout(dropout)

  def forward(
      self,
      tgt: torch.Tensor,
      query_memory: torch.Tensor,
      context_S: torch.Tensor | None,
      context_z: torch.Tensor | None,
      tgt_mask: torch.Tensor | None = None,
      tgt_is_causal: bool = False,
      memory_key_padding_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Forward pass through the four sub-layers.

    Args:
      tgt: (B, L_dec, d_model) — decoder input.
      query_memory: (B, L_enc, d_model) — encoder output for the query.
      context_S: (B, H, m, d_v) — context state accumulator, or ``None`` if no
        context pairs are available (k=0).
      context_z: (B, H, m) — context normalizer, or ``None``.
      tgt_mask: Optional causal mask for self-attention.
      tgt_is_causal: Whether ``tgt_mask`` encodes a causal constraint.
      memory_key_padding_mask: Padding mask for ``query_memory``.

    Returns:
      (B, L_dec, d_model) — layer output.
    """
    x = tgt

    # 1. Pre-norm causal self-attention.
    x2 = self.norm1(x)
    x = (
        x
        + self.self_attn(
            x2,
            x2,
            x2,
            attn_mask=tgt_mask,
            is_causal=tgt_is_causal,
        )[0]
    )

    # 2. Pre-norm softmax cross-attention to query encoder memory.
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

    # 3. Pre-norm linear cross-attention to context state S.
    #    Skipped when no context pairs have been provided (k=0).
    if context_S is not None and context_z is not None:
      x2 = self.norm3(x)
      x = x + self.context_cross_attn(x2, context_S, context_z)

    # 4. Pre-norm FFN.
    x2 = self.norm4(x)
    x = x + self.linear2(self.ffn_dropout(F.relu(self.linear1(x2))))

    return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class IncrementalEncoderDecoder(nn.Module):
  """Encoder-decoder with incremental linear-attention context state.

  This model extends the standard ``EncoderDecoder`` pattern with a third
  cross-attention sub-layer in each decoder layer that attends to an
  accumulated context state **S**.  The state can be built from a batch of
  (x, y) context pairs and updated incrementally in O(L·d²) per pair.

  Usage::

      model = IncrementalEncoderDecoder(...)
      # Build context from k pairs.
      state = model.build_context_state(ctx_src_list, ctx_y_list)
      # Optionally add one more pair.
      state = model.update_context_state(state, new_src, new_y)
      # Training forward pass.
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
      decoder_dropout: float = 0.0,
      encoder_type: encoders.EncoderType = encoders.EncoderType.VANILLA,
      additional_encoder_kwargs: dict[str, Any] | None = None,
      num_features: int | None = None,
  ):
    """Initializes the IncrementalEncoderDecoder.

    Args:
      encoder_vocab_size: Vocabulary size for the encoder.
      decoder_vocab_size: Vocabulary size for the decoder.
      encoder_pad_idx: Padding token index used by the encoder.
      max_encoder_len: Maximum source sequence length.
      max_decoder_len: Maximum target sequence length.
      d_model: Model hidden dimension.
      num_encoder_layers: Number of encoder layers.
      num_decoder_layers: Number of decoder layers.
      decoder_dropout: Dropout rate for the FFN inside each decoder layer.
      encoder_type: Encoder architecture to use.
      additional_encoder_kwargs: Extra kwargs forwarded to the encoder factory.
      num_features: Feature-map dimension for the linear cross-attention
        sub-layers.  Defaults to ``d_model``.
    """
    super().__init__()
    self.encoder_pad_idx = encoder_pad_idx

    # Shared encoder (unchanged from EncoderDecoder).
    self.encoder = encoder_type.make(
        vocab_size=encoder_vocab_size,
        d_model=d_model,
        num_layers=num_encoder_layers,
        max_encoder_len=max_encoder_len,
        **(additional_encoder_kwargs or {}),
    )

    hidden_dim = self.encoder.hidden_dim

    # Shared decoder token embedding (also used to embed context y values).
    self.tgt_tok_emb = nn.Embedding(decoder_vocab_size, hidden_dim)
    self.decoder_positional_encoding = _PositionalEncoding(
        hidden_dim,
        max_len=max_decoder_len,
    )

    # Custom decoder layers with the extra linear cross-attention sub-layer.
    self.decoder_layers = nn.ModuleList([
        _IncrementalDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=4 * hidden_dim,
            num_features=num_features,
            dropout=decoder_dropout,
        )
        for _ in range(num_decoder_layers)
    ])

    self.generator = nn.Linear(hidden_dim, decoder_vocab_size)

  # ---- Encoder interface ---------------------------------------------------

  def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes the source (query) sequence.

    Args:
      src: (B, L) int tensor of encoder token ids.

    Returns:
      A tuple ``(memory, src_padding_mask)`` where
        memory: (B, L, d_model) encoder output.
        src_padding_mask: (B, L) bool mask (True = pad).
    """
    src_padding_mask = src == self.encoder_pad_idx
    with nn.attention.sdpa_kernel(SPD_BACKENDS):
      memory = self.encoder(src, src_key_padding_mask=src_padding_mask)
    return memory, src_padding_mask

  # ---- Context state construction ------------------------------------------

  def build_context_state(
      self,
      context_src_list: list[torch.Tensor],
      context_y_token_ids_list: list[torch.Tensor],
  ) -> ContextState:
    """Builds the context state S from a list of (x, y) pairs.

    For each pair the x is encoded through the shared encoder and the y token
    ids are embedded through the shared decoder embedding table.  The
    concatenated tokens are then projected and accumulated into the per-layer
    linear-attention states.

    Args:
      context_src_list: List of k tensors, each (B, L_i) — encoder input token
        ids for each context x_i.
      context_y_token_ids_list: List of k tensors, each (B, T_yi) — decoder
        token ids for each context y_i.

    Returns:
      A ``ContextState`` containing accumulated S and z for every decoder
      layer.
    """
    num_layers = len(self.decoder_layers)
    # Initialize accumulators to None; lazily set on first pair.
    S_accum: list[torch.Tensor | None] = [None] * num_layers
    z_accum: list[torch.Tensor | None] = [None] * num_layers

    for src_i, y_ids_i in zip(context_src_list, context_y_token_ids_list):
      # Encode context x_i.
      padding_mask_i = src_i == self.encoder_pad_idx
      with nn.attention.sdpa_kernel(SPD_BACKENDS):
        enc_x = self.encoder(src_i, src_key_padding_mask=padding_mask_i)

      # Embed context y_i using the shared decoder embedding.
      emb_y = self.tgt_tok_emb(y_ids_i)  # (B, T_yi, d)

      # Concatenate along sequence dimension.
      context_tokens = torch.cat([enc_x, emb_y], dim=1)  # (B, L_i+T_yi, d)

      # Build combined padding mask: encoder pad positions are True,
      # y-embedding positions are never padded (False).
      y_padding = torch.zeros(
          y_ids_i.shape[0],
          y_ids_i.shape[1],
          dtype=torch.bool,
          device=src_i.device,
      )
      combined_padding_mask = torch.cat(
          [padding_mask_i, y_padding],
          dim=1,
      )  # (B, L_i + T_yi)

      # Accumulate state for each decoder layer.
      for l, layer in enumerate(self.decoder_layers):
        S_l, z_l = layer.context_cross_attn.build_state(
            context_tokens,
            padding_mask=combined_padding_mask,
        )
        if S_accum[l] is None:
          S_accum[l] = S_l
          z_accum[l] = z_l
        else:
          S_accum[l] = S_accum[l] + S_l
          z_accum[l] = z_accum[l] + z_l

    # If no context pairs were provided, return zero-initialized state so
    # downstream code can still index into the lists.
    if S_accum[0] is None:
      raise ValueError(
          "context_src_list must contain at least one context pair.  "
          "For k=0 context, pass context_state=None to forward()."
      )

    return ContextState(S=S_accum, z=z_accum)  # type: ignore[arg-type]

  def update_context_state(
      self,
      state: ContextState,
      new_src: torch.Tensor,
      new_y_token_ids: torch.Tensor,
  ) -> ContextState:
    """Incrementally adds one (x, y) pair to an existing context state.

    This is O(L · d²) where L is the combined length of the new pair's
    encoder output and y embedding.

    Args:
      state: Existing ``ContextState`` to update.
      new_src: (B, L_new) encoder input token ids for the new x.
      new_y_token_ids: (B, T_y_new) decoder token ids for the new y.

    Returns:
      Updated ``ContextState`` with the new pair accumulated.
    """
    # Encode the new x.
    padding_mask = new_src == self.encoder_pad_idx
    with nn.attention.sdpa_kernel(SPD_BACKENDS):
      enc_x = self.encoder(new_src, src_key_padding_mask=padding_mask)

    emb_y = self.tgt_tok_emb(new_y_token_ids)
    context_tokens = torch.cat([enc_x, emb_y], dim=1)

    y_padding = torch.zeros(
        new_y_token_ids.shape[0],
        new_y_token_ids.shape[1],
        dtype=torch.bool,
        device=new_src.device,
    )
    combined_padding_mask = torch.cat([padding_mask, y_padding], dim=1)

    new_S_list: list[torch.Tensor] = []
    new_z_list: list[torch.Tensor] = []
    for l, layer in enumerate(self.decoder_layers):
      S_l, z_l = layer.context_cross_attn.build_state(
          context_tokens,
          padding_mask=combined_padding_mask,
      )
      new_S_list.append(state.S[l] + S_l)
      new_z_list.append(state.z[l] + z_l)

    return ContextState(S=new_S_list, z=new_z_list)

  # ---- Training forward pass -----------------------------------------------

  def forward(
      self,
      src: torch.Tensor,
      tgt_input: torch.Tensor,
      context_state: ContextState | None = None,
  ) -> torch.Tensor:
    """Training forward pass.

    Args:
      src: (B, L_src) encoder input token ids for the query.
      tgt_input: (B, L_tgt) decoder input token ids (teacher-forced).
      context_state: Optional ``ContextState`` built from context (x, y) pairs.
        When ``None``, the linear cross-attention sub-layers are skipped (k=0
        context).

    Returns:
      (B, L_tgt, vocab) logits over the decoder vocabulary.
    """
    memory, memory_padding_mask = self.encode(src)
    tgt = self.decoder_positional_encoding(self.tgt_tok_emb(tgt_input))
    tgt_mask = self._get_tgt_mask(tgt_input)

    x = tgt
    for l, layer in enumerate(self.decoder_layers):
      S_l = context_state.S[l] if context_state is not None else None
      z_l = context_state.z[l] if context_state is not None else None
      x = layer(
          x,
          memory.to(dtype=self.tgt_tok_emb.weight.dtype),
          S_l,
          z_l,
          tgt_mask=tgt_mask,
          tgt_is_causal=True,
          memory_key_padding_mask=memory_padding_mask,
      )
    return self.generator(x)

  # ---- Autoregressive decoding ---------------------------------------------

  def next_token_logits(
      self,
      current_tgt_seq: torch.Tensor,
      memory: torch.Tensor,
      memory_key_padding_mask: torch.Tensor,
      context_state: ContextState | None = None,
  ) -> torch.Tensor:
    """Computes logits for the next token during autoregressive generation.

    Args:
      current_tgt_seq: (B, T) decoder token ids generated so far.
      memory: (B, L_src, d_model) encoder output for the query.
      memory_key_padding_mask: (B, L_src) padding mask for the encoder output.
      context_state: Optional ``ContextState``.

    Returns:
      (B, vocab) logits for the next token position.
    """
    tgt = self.decoder_positional_encoding(self.tgt_tok_emb(current_tgt_seq))
    tgt_mask = self._get_tgt_mask(current_tgt_seq)

    x = tgt
    for l, layer in enumerate(self.decoder_layers):
      S_l = context_state.S[l] if context_state is not None else None
      z_l = context_state.z[l] if context_state is not None else None
      x = layer(
          x,
          memory.to(dtype=self.tgt_tok_emb.weight.dtype),
          S_l,
          z_l,
          tgt_mask=tgt_mask,
          tgt_is_causal=True,
          memory_key_padding_mask=memory_key_padding_mask,
      )
    return self.generator(x[:, -1, :])

  # ---- Utilities -----------------------------------------------------------

  def _get_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor | None:
    return nn.Transformer.generate_square_subsequent_mask(
        tgt.size(1),
        device=tgt.device,
        dtype=torch.bool,
    )
