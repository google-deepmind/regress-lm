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

import functools
from regress_lm import core
from regress_lm.pytorch import t5gemma_model
import torch
from absl.testing import absltest


class T5gemmaModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = t5gemma_model.T5GemmaModelConfig(
        model_name="google/t5gemma-s-s-prefixlm",
        max_input_len=10,
        max_decode_len=10,
        y_to_str_fn=functools.partial(
            t5gemma_model.default_y_to_str_fn, precision=1
        ),
    )
    self.model = self.cfg.make_model()

  def test_convert_examples(self):
    examples = [core.Example(x="a s d", y=1.0), core.Example(x="z x c", y=0.0)]
    batch = self.model.converter.convert_examples(examples)
    self.assertEqual(batch["input_ids"].shape, (2, 10))
    self.assertEqual(batch["attention_mask"].shape, (2, 10))
    self.assertEqual(batch["labels"].shape, (2, 10))

    # x = "a s d"
    self.assertEqual(
        batch["input_ids"][0].tolist(), [235250, 485, 499, 0, 0, 0, 0, 0, 0, 0]
    )

    # y = 1.0 (as string "1.0e+00") + <eos> token (token ID = 1) + -100 paddings
    self.assertEqual(
        batch["labels"][0].tolist(),
        [235274, 235265, 235276, 235249, 235340, 235276, 235276, 1, -100, -100],
    )

  def test_loss_and_metrics(self):
    examples = [core.Example(x="hello", y=1.0), core.Example(x="world", y=0.0)]
    batch = self.model.converter.convert_examples(examples)
    losses_per_example, _ = self.model.compute_losses_and_metrics(batch)
    self.assertEqual(losses_per_example.shape, (2,))
    self.assertAlmostEqual(losses_per_example.mean().item(), 4.9336, 3)

  def test_log_prob(self):
    examples = [core.Example(x="hello", y=1.0), core.Example(x="world", y=0.0)]
    batch = self.model.converter.convert_examples(examples)
    log_probs = self.model.log_prob(batch)
    self.assertEqual(log_probs.shape, (2,))
    self.assertAlmostEqual(log_probs[0].item(), -40.6720, 3)
    self.assertAlmostEqual(log_probs[1].item(), -38.2665, 3)

  @absltest.skip("T5Gemma sometimes not up to date with HF")
  def test_decode(self):
    batch = self.model.converter.convert_inputs(
        [core.ExampleInput(x="a s d"), core.ExampleInput(x="z x c")]
    )

    torch.manual_seed(42)
    decoded_ids, string_outputs = self.model.decode(batch, num_samples=2)

    self.assertLessEqual(decoded_ids.shape[2], 10)  # Max decode len.
    self.assertEqual(string_outputs.shape, (2, 2))
    self.assertEqual(string_outputs[0, 0], "  a")
    self.assertEqual(string_outputs[0, 1], " have this one's\n\nThe")


if __name__ == "__main__":
  absltest.main()
