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

"""Tests for torch-specific data utils, including DataLoader integration."""

from regress_lm import core
from regress_lm.pytorch import data_utils
import torch
from torch import utils
from absl.testing import absltest


class DataLoaderIntegrationTest(absltest.TestCase):

  def test_dataloader_integration(self):
    """Verifies DataLoader pipeline usage."""

    raw_examples = [
        core.Example(x="Prompt A", y=1.0),
        core.Example(x="Prompt B", y=2.0),
        core.Example(x="Prompt C", y=3.0),
        core.Example(x="Prompt D", y=4.0),
    ]

    # Create the lightweight Dataset.
    dataset = data_utils.ExampleDataset(raw_examples)

    # Mock collate function that simulates `model.convert_examples`.
    def mock_collate_fn(
        examples: list[core.Example],
    ) -> dict[str, torch.Tensor]:
      batch_size = len(examples)
      input_tensors = [
          torch.tensor([i, i + 1, i + 2]) for i in range(batch_size)
      ]
      target_tensors = [torch.tensor([ex.y]) for ex in examples]
      return {
          "input_tensor": torch.stack(input_tensors),
          "target_tensor": torch.stack(target_tensors),
      }

    loader = utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=mock_collate_fn,
        num_workers=0,  # Avoid parallelism in testing.
    )

    batches = list(loader)
    self.assertLen(batches, 2)

    # --- Inspect batch 0 ---
    batch_0 = batches[0]
    self.assertIsInstance(batch_0, dict)
    self.assertCountEqual(batch_0.keys(), ["input_tensor", "target_tensor"])

    # Check shapes.
    self.assertEqual(batch_0["input_tensor"].shape, (2, 3))
    self.assertEqual(batch_0["target_tensor"].shape, (2, 1))

    # Check content. It should correspond to the first two raw examples.
    expected_input_0 = torch.tensor([[0, 1, 2], [1, 2, 3]])
    expected_target_0 = torch.tensor([[1.0], [2.0]])
    torch.testing.assert_close(batch_0["input_tensor"], expected_input_0)
    torch.testing.assert_close(batch_0["target_tensor"], expected_target_0)

    # --- Inspect batch 1 ---
    batch_1 = batches[1]
    self.assertEqual(batch_1["input_tensor"].shape, (2, 3))
    self.assertEqual(batch_1["target_tensor"].shape, (2, 1))

    # Check content. It should correspond to the next two raw examples.
    expected_input_1 = torch.tensor([[0, 1, 2], [1, 2, 3]])
    expected_target_1 = torch.tensor([[3.0], [4.0]])
    torch.testing.assert_close(batch_1["input_tensor"], expected_input_1)
    torch.testing.assert_close(batch_1["target_tensor"], expected_target_1)


if __name__ == "__main__":
  absltest.main()
