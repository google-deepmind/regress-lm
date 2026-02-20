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

import math
from typing import Any
from regress_lm.pytorch import encoders
import torch
from torch import nn


# Backends attempted in order.
SPD_BACKENDS = [
    nn.attention.SDPBackend.FLASH_ATTENTION,
    nn.attention.SDPBackend.CUDNN_ATTENTION,
    nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    nn.attention.SDPBackend.MATH,  # Last resort, materializes whole matrix.
]


class _PositionalEncoding(nn.Module):
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
      num_encoder_layers: int,
      num_decoder_layers: int,
      decoder_dropout: float = 0.0,
      # encoder args
      encoder_type: encoders.EncoderType = encoders.EncoderType.VANILLA,
      additional_encoder_kwargs: dict[str, Any] | None = None,
  ):
    super().__init__()
    self.encoder_pad_idx = encoder_pad_idx
    self.encoder = encoder_type.make(
        vocab_size=encoder_vocab_size,
        d_model=d_model,
        num_layers=num_encoder_layers,
        max_encoder_len=max_encoder_len,
        **(additional_encoder_kwargs or {}),
    )

    # We use the hidden_dim of the encoder for the decoder.
    self.tgt_tok_emb = nn.Embedding(decoder_vocab_size, self.encoder.hidden_dim)
    self.decoder_positional_encoding = _PositionalEncoding(
        self.encoder.hidden_dim,
        max_len=max_decoder_len,
        dropout=decoder_dropout,
    )
    decoder_layer = nn.TransformerDecoderLayer(
        self.encoder.hidden_dim,
        nhead=8,
        dim_feedforward=4 * self.encoder.hidden_dim,
        dropout=decoder_dropout,
        batch_first=True,
        norm_first=True,
    )
    self.decoder = nn.TransformerDecoder(
        decoder_layer, num_layers=num_decoder_layers
    )
    self.generator = nn.Linear(self.encoder.hidden_dim, decoder_vocab_size)

  def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
    src_padding_mask = src == self.encoder_pad_idx

    with nn.attention.sdpa_kernel(SPD_BACKENDS):
      memory = self.encoder(src, src_key_padding_mask=src_padding_mask)
      decoder_output = self.decoder(
          tgt=self.decoder_positional_encoding(self.tgt_tok_emb(tgt_input)),
          memory=memory.to(dtype=self.tgt_tok_emb.weight.dtype),
          tgt_mask=self._get_tgt_mask(tgt_input),
          tgt_is_causal=True,
          memory_key_padding_mask=src_padding_mask,
      )
    return self.generator(decoder_output)

  def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes the source sequence."""
    src_padding_mask = src == self.encoder_pad_idx
    with nn.attention.sdpa_kernel(SPD_BACKENDS):
      memory = self.encoder(src, src_key_padding_mask=src_padding_mask)
    return memory, src_padding_mask

  def next_token_logits(
      self,
      current_tgt_seq: torch.Tensor,
      memory: torch.Tensor,
      memory_key_padding_mask: torch.Tensor,
  ) -> torch.Tensor:
    """Decodes one step using the standard decoder."""
    tgt = self.decoder_positional_encoding(self.tgt_tok_emb(current_tgt_seq))

    with nn.attention.sdpa_kernel(SPD_BACKENDS):
      decoder_output_all_steps = self.decoder(
          tgt=tgt,
          memory=memory.to(dtype=self.tgt_tok_emb.weight.dtype),
          tgt_mask=self._get_tgt_mask(current_tgt_seq),
          tgt_is_causal=True,
          memory_key_padding_mask=memory_key_padding_mask,
      )
    return self.generator(decoder_output_all_steps[:, -1, :])

  def _get_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor | None:
    """Returns explicit target mask if needed (e.g. on CPUs and tests)."""
    # TorchInductor will ignore this after compilation, allowing speedups.
    return nn.Transformer.generate_square_subsequent_mask(
        tgt.size(1), device=tgt.device, dtype=torch.bool
    )
