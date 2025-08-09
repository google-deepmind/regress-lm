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

from regress_lm.models.pytorch import encoders
import torch

from absl.testing import absltest
from absl.testing import parameterized


class MambaEncoderTest(absltest.TestCase):

  def setUp(self):
    """Set up a small, deterministic MambaEncoder for all tests."""
    super().setUp()
    self.d_model = 16
    self.encoder = encoders.MambaEncoder(
        d_model=self.d_model, num_layers=2, headdim=16
    )

  def test_forward_shape(self):
    """Tests that the encoder output has the correct shape for unpadded input."""
    batch_size = 4
    seq_len = 50
    # Input tensor (simulating embedded tokens)
    src_emb = torch.randn(batch_size, seq_len, self.d_model)

    # Forward pass without any padding mask
    with torch.no_grad():
      output = self.encoder(src_emb, src_key_padding_mask=None)

    # Assert that the output shape is identical to the input shape
    self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))


class PerformerEncoderTest(parameterized.TestCase):

  @parameterized.parameters(("softmax",), ("relu",))
  def test_forward_shape(self, kernel_name: str):
    """Tests that the encoder output has the correct shape."""
    batch_size = 4
    seq_len = 50
    d_model = 16

    src_emb = torch.randn(batch_size, seq_len, d_model)
    encoder = encoders.PerformerEncoder(
        d_model=d_model, num_layers=2, kernel_name=kernel_name  # pytype: disable=wrong-arg-types
    )
    output = encoder(src_emb, src_key_padding_mask=None)

    self.assertEqual(output.shape, (batch_size, seq_len, d_model))
    self.assertFalse(
        torch.isnan(output).any(), "Output should not contain NaNs."
    )


if __name__ == "__main__":
  absltest.main()
