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

from typing import Any
from regress_lm.pytorch import decoders
from regress_lm.pytorch import encoders
import torch
from torch import nn

autocast = torch.amp.autocast

# Backends attempted in order.
SPD_BACKENDS = [
    nn.attention.SDPBackend.FLASH_ATTENTION,
    nn.attention.SDPBackend.CUDNN_ATTENTION,
    nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    nn.attention.SDPBackend.MATH,  # Last resort, materializes whole matrix.
]


class EncoderDecoder(nn.Module):
  """Encoder-Decoder model in PyTorch."""

  def __init__(
      self,
      encoder_vocab_size: int,
      decoder_vocab_size: int,
      encoder_pad_idx: int,
      max_encoder_len: int,
      d_model: int,
      num_encoder_layers: int,
      num_decoder_layers: int,
      decoder_dropout: float = 0.0,
      # encoder args
      encoder_type: encoders.EncoderType = encoders.EncoderType.VANILLA,
      additional_encoder_kwargs: dict[str, Any] | None = None,
      try_bf16: bool = True,  # Only on cuda.
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
    self.use_bf16 = torch.cuda.is_available() and try_bf16

    # We use the hidden_dim of the encoder for the decoder.
    d_enc = self.encoder.hidden_dim
    self.tgt_tok_emb = nn.Embedding(decoder_vocab_size, d_enc)
    self.decoder = decoders.Decoder(d_enc, num_decoder_layers, decoder_dropout)
    self.generator = nn.Linear(d_enc, decoder_vocab_size)

  def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
    src_padding_mask = src == self.encoder_pad_idx

    with autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
      with nn.attention.sdpa_kernel(SPD_BACKENDS):
        memory = self.encoder(src, src_key_padding_mask=src_padding_mask)
        tgt = self.tgt_tok_emb(tgt_input)
        out = self.decoder(tgt, memory, src_padding_mask)
    return self.generator(out.float())

  def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes the source sequence."""
    src_padding_mask = src == self.encoder_pad_idx
    with autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
      with nn.attention.sdpa_kernel(SPD_BACKENDS):
        memory = self.encoder(src, src_key_padding_mask=src_padding_mask)
    return memory, src_padding_mask

  def next_token_logits(
      self,
      current_tgt_seq: torch.Tensor,
      memory: torch.Tensor,
      memory_key_padding_mask: torch.Tensor,
  ) -> torch.Tensor:
    """Decodes one step. (B * S, T) targets may share (B, L, D) memory."""
    with autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
      with nn.attention.sdpa_kernel(SPD_BACKENDS):
        tgt = self.tgt_tok_emb(current_tgt_seq)
        out = self.decoder(tgt, memory, memory_key_padding_mask)
    return self.generator(out[:, -1, :].float())
