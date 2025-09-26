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

"""Tests for inference functions."""

from unittest import mock
from regress_lm import core
from regress_lm.pytorch import inference
import torch
from absl.testing import absltest


class RAFTInferenceTest(absltest.TestCase):
  """Tests for the RAFTInferenceFn."""

  def test_correct_mean_outputs(self):
    """Verifies weighted average is calculated correctly."""
    mock_model = mock.MagicMock(spec=core.Model)
    grid_points = [10.0, 20.0, 30.0]
    raft_fn = inference.RAFTInferenceFn(
        model=mock_model, grid_points=grid_points
    )

    # Test with a single example
    single_log_probs = torch.log(torch.tensor([0.1, 0.2, 0.7]))
    mock_model.log_prob.return_value = single_log_probs

    # Expected mean: (10 * 0.1) + (20 * 0.2) + (30 * 0.7) = 26.0
    result_single = raft_fn.infer([core.ExampleInput(x='prompt 1')])
    torch.testing.assert_close(result_single, torch.tensor([26.0]))

    # Test with a batch of two examples
    batch_log_probs = torch.log(
        torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.8, 0.1]]).flatten()
    )
    mock_model.log_prob.return_value = batch_log_probs

    # Expected mean 1: (10*0.6) + (20*0.3) + (30*0.1) = 6+6+3 = 15.0
    # Expected mean 2: (10*0.1) + (20*0.8) + (30*0.1) = 1+16+3 = 20.0
    result_batch = raft_fn.infer(
        [core.ExampleInput(x='prompt 1'), core.ExampleInput(x='prompt 2')]
    )
    torch.testing.assert_close(result_batch, torch.tensor([15.0, 20.0]))


if __name__ == '__main__':
  absltest.main()
