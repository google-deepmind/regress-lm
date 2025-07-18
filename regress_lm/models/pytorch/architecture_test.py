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

"""Tests for the PyTorch architecture."""

from regress_lm.models.pytorch import architecture
import torch
from absl.testing import absltest


class ArchitectureTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.encoder_vocab_size = 10
    self.decoder_vocab_size = 5
    self.encoder_pad_idx = 0
    self.decoder_pad_idx = 0
    self.max_encoder_len = 15
    self.max_decoder_len = 6 + 1

    self.src_seq_len = 10  # Less than max_encoder_len

    self.d_model = 32

    self.model = architecture.EncoderDecoder(
        encoder_vocab_size=self.encoder_vocab_size,
        decoder_vocab_size=self.decoder_vocab_size,
        encoder_pad_idx=self.encoder_pad_idx,
        max_encoder_len=self.max_encoder_len,
        max_decoder_len=self.max_decoder_len,
        d_model=self.d_model,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.1,
    )

    # Dummy data parameters
    self.batch_size = 2

  def test_forward(self):
    src = torch.randint(
        1, self.encoder_vocab_size, (self.batch_size, self.src_seq_len)
    )
    # Add some padding to src
    src[0, -2:] = self.encoder_pad_idx
    src[1, -1:] = self.encoder_pad_idx

    tgt_input = torch.randint(
        1, self.decoder_vocab_size, (self.batch_size, self.max_decoder_len)
    )
    # Add some padding to tgt_input
    tgt_input[0, -1:] = self.decoder_pad_idx

    output_logits = self.model(src, tgt_input)

    self.assertEqual(
        output_logits.shape,
        (self.batch_size, self.max_decoder_len, self.decoder_vocab_size),
    )

  def test_encode(self):
    src = torch.randint(
        1, self.encoder_vocab_size, (self.batch_size, self.src_seq_len)
    )
    src[0, -3:] = self.encoder_pad_idx  # Add padding

    memory, memory_key_padding_mask = self.model.encode(src)

    self.assertEqual(
        memory.shape, (self.batch_size, self.src_seq_len, self.d_model)
    )
    self.assertEqual(
        memory_key_padding_mask.shape, (self.batch_size, self.src_seq_len)
    )

    # Check if padding mask reflects actual padding
    self.assertTrue(torch.all(memory_key_padding_mask[0, -3:]).item())
    self.assertFalse(torch.any(memory_key_padding_mask[0, :-3]).item())

  def test_next_token_logits(self):
    src = torch.randint(
        1, self.encoder_vocab_size, (self.batch_size, self.src_seq_len)
    )
    memory, memory_key_padding_mask = self.model.encode(src)

    # Simulate decoding one step with just a start token
    current_tgt_len = 1
    current_tgt_seq = torch.full(
        (self.batch_size, current_tgt_len),
        self.decoder_pad_idx,
        dtype=torch.long,
    )

    next_token_logits = self.model.next_token_logits(
        current_tgt_seq, memory, memory_key_padding_mask
    )

    self.assertEqual(
        next_token_logits.shape, (self.batch_size, self.decoder_vocab_size)
    )

    # Simulate decoding a few more steps
    current_tgt_len = 3
    current_tgt_seq = torch.randint(
        1, self.decoder_vocab_size, (self.batch_size, current_tgt_len)
    )
    # Ensure it starts with start token
    current_tgt_seq[:, 0] = self.decoder_pad_idx

    next_token_logits_multi = self.model.next_token_logits(
        current_tgt_seq, memory, memory_key_padding_mask
    )
    self.assertEqual(
        next_token_logits_multi.shape,
        (self.batch_size, self.decoder_vocab_size),
    )

  def test_basic_autoregressive_decode_loop(self):
    batch_size = 1
    src = torch.randint(
        batch_size, self.encoder_vocab_size, (batch_size, self.src_seq_len)
    )
    memory, memory_key_padding_mask = self.model.encode(src)

    # Begin with the start (padding) token.
    current_tgt_tokens = torch.LongTensor([[self.decoder_pad_idx]])

    max_decode_steps = 5
    decoded_ids = [self.decoder_pad_idx]

    for _ in range(max_decode_steps):
      # (1, num_decoder_tokens)
      next_token_logits = self.model.next_token_logits(
          current_tgt_tokens, memory, memory_key_padding_mask
      )

      # Get the token ID
      predicted_token_id = torch.argmax(next_token_logits, dim=1).item()
      decoded_ids.append(predicted_token_id)

      # Append predicted token to current_tgt_tokens for next step
      current_tgt_tokens = torch.cat(
          [current_tgt_tokens, torch.LongTensor([[predicted_token_id]])],
          dim=1,
      )

    self.assertGreater(len(decoded_ids), 1)
    self.assertTrue(
        all(0 <= token_id < self.decoder_vocab_size for token_id in decoded_ids)
    )


if __name__ == "__main__":
  absltest.main()
