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

import math

from regress_lm.models.pytorch import encoders
import torch
from torch.nn import functional as F

from absl.testing import absltest
from absl.testing import parameterized


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

  def test_forward_with_padding_invariance(self):
    """Tests that padding is handled correctly by the attention mask.

    The output for real tokens should be the same regardless of padding.
    """
    batch_size = 2
    max_seq_len = 10
    d_model = 16

    torch.manual_seed(42)
    seq1 = torch.randn(1, 5, d_model)  # Length 5
    seq2 = torch.randn(1, 8, d_model)  # Length 8

    encoder = encoders.PerformerEncoder(d_model=d_model, num_layers=2)

    # --- Run without padding (the ground truth) ---
    with torch.no_grad():
      output1_unpadded = encoder(seq1, src_key_padding_mask=None)
      output2_unpadded = encoder(seq2, src_key_padding_mask=None)

    # --- Run with padding ---
    # Now, we combine them into a single batch, padding seq1 to the max length.
    padded_src_emb = torch.zeros(batch_size, max_seq_len, d_model)
    padded_src_emb[0, :5, :] = seq1
    padded_src_emb[1, :8, :] = seq2

    # Create the corresponding padding mask (True where it's padded)
    src_key_padding_mask = torch.tensor(
        [
            [False, False, False, False, False, True, True, True, True, True],
            [False, False, False, False, False, False, False, False, True, True],  # pylint: disable=line-too-long
        ],
        dtype=torch.bool,
    )

    with torch.no_grad():
      output_padded = encoder(padded_src_emb, src_key_padding_mask)

    # Check that the output for the non-padded parts is very close to the
    # unpadded runs. Due to floating point math, a small tolerance is needed.
    self.assertTrue(
        torch.allclose(output_padded[0, :5, :], output1_unpadded, atol=1e-6),
        "Content of sequence 1 should be unchanged by padding.",
    )
    self.assertTrue(
        torch.allclose(output_padded[1, :8, :], output2_unpadded, atol=1e-6),
        "Content of sequence 2 should be unchanged by padding.",
    )
    # Note: Unlike Mamba, the padded positions in the output of an attention
    # layer are NOT expected to be zero. They are valid outputs, just computed
    # without looking at the padded keys. The critical test is the invariance
    # of the non-padded tokens' outputs.


class FavorAttentionTest(parameterized.TestCase):
  """Low-level unit tests for the FAVOR+ attention mechanism, ported from JAX.

  These tests verify the mathematical correctness against standard softmax
  attention.
  """

  def test_favor_plus_attention_with_no_padding(self):
    """Tests accuracy of FAVOR+ (softmax kernel) vs exact softmax attention.

    With a large number of random features, the approximation should be close.
    """
    head_dim = 8
    v_dim = 10
    batch_size = 1
    length = 2
    num_heads = 1
    num_features = 10000

    torch.manual_seed(0)

    # Create random tensors
    query = torch.randn(batch_size, length, num_heads, head_dim)
    key = torch.randn(batch_size, length, num_heads, head_dim)
    value = torch.randn(batch_size, length, num_heads, v_dim)

    projection_matrix = encoders._gaussian_orthogonal_random_matrix(
        num_features, head_dim
    )
    favorplus_result = encoders._favor_attention(
        query,
        key,
        value,
        kernel_fn=encoders._exp_softmax_kernel_transformation,
        projection_matrix=projection_matrix,
        inputs_mask=None,
    )

    # --- 2. Calculate Exact Softmax Attention ---
    query_th = query.permute(0, 2, 1, 3)  # B, L, H, D -> B, H, L, D
    key_th = key.permute(0, 2, 1, 3)
    value_th = value.permute(0, 2, 1, 3)
    exact_result_th = F.scaled_dot_product_attention(query_th, key_th, value_th)
    # B, H, L, D -> B, L, H, D
    exact_result = exact_result_th.permute(0, 2, 1, 3)

    # Results should be close.
    self.assertTrue(torch.allclose(favorplus_result, exact_result, atol=0.7))

  def test_favor_plus_attention_with_padding(self):
    """Tests the approximation of FAVOR+ with an explicit attention mask."""

    head_dim = 8
    v_dim = 10
    batch_size = 1
    length = 2
    num_heads = 1
    num_features = 10000

    torch.manual_seed(0)

    query = torch.randn(batch_size, length, num_heads, head_dim)
    key = torch.randn(batch_size, length, num_heads, head_dim)
    value = torch.randn(batch_size, length, num_heads, v_dim)

    # inputs_mask is True for real tokens, False for padding
    inputs_mask = torch.tensor([[True, False]], dtype=torch.bool)
    projection_matrix = encoders._gaussian_orthogonal_random_matrix(
        num_features, head_dim
    )
    favorplus_result = encoders._favor_attention(
        query,
        key,
        value,
        kernel_fn=encoders._exp_softmax_kernel_transformation,
        projection_matrix=projection_matrix,
        inputs_mask=inputs_mask,
    )

    # Permute to (B, H, L, D) for standard attention calculations
    query_th = query.permute(0, 2, 1, 3)
    key_th = key.permute(0, 2, 1, 3)
    value_th = value.permute(0, 2, 1, 3)
    # (B, H, T, D) @ (B, H, D, S) -> (B, H, T, S)
    attention_scores = torch.matmul(
        query_th, key_th.transpose(-2, -1)
    ) / math.sqrt(head_dim)

    # Create the additive mask
    # We want to add a large negative number where inputs_mask is False
    # Mask is (B, L), we need to broadcast to (B, H, T, S)
    # Here, T=L, S=L
    additive_mask = torch.zeros_like(attention_scores)
    additive_mask.masked_fill_(
        ~inputs_mask.unsqueeze(1).unsqueeze(2), float("-inf")
    )

    # Apply the mask and then softmax
    attention_scores = attention_scores + additive_mask
    attention_probs = F.softmax(attention_scores, dim=-1)

    # Apply attention probs to values
    exact_result_th = torch.matmul(attention_probs, value_th)
    exact_result = exact_result_th.permute(0, 2, 1, 3)  # Back to (B, L, H, D)

    # Compare results.
    self.assertTrue(
        torch.allclose(
            favorplus_result[:, 0, ...], exact_result[:, 0, ...], atol=0.7
        ),
        "Mismatch on non-padded token. \nFavor:"
        f" {favorplus_result[:, 0, ...]} \nExact: {exact_result[:, 0, ...]}",
    )


if __name__ == "__main__":
  absltest.main()
