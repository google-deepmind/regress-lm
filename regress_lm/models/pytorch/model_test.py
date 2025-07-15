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

"""Tests for the PyTorch model."""

import numpy as np
from regress_lm import core
from regress_lm import tokenizers
from regress_lm import vocabs
from regress_lm.models.pytorch import model as pytorch_model
import torch
from torch import optim
from absl.testing import absltest


class ModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.encoder_vocab = vocabs.BasicEnglishVocab(["hello", "world"])
    self.decoder_vocab = vocabs.DecoderVocab(tokenizers.P10Tokenizer())
    architecture_kwargs = dict(
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.1,
    )
    self.model = pytorch_model.PyTorchModel(
        encoder_vocab=self.encoder_vocab,
        decoder_vocab=self.decoder_vocab,
        max_input_len=4,
        **architecture_kwargs
    )
    self.optimizer = optim.Adafactor(
        filter(lambda p: p.requires_grad, self.model.parameters()), lr=1.0
    )

  def test_convert(self):
    batch = self.model.convert_examples(
        [core.Example(x="hello", y=1.0), core.Example(x="world", y=2.0)]
    )

    np.testing.assert_array_equal(
        batch["encoder_input"], [[2, 0, 0, 0], [3, 0, 0, 0]]
    )

    np.testing.assert_array_equal(
        batch["decoder_input"], [[0, 1, 4, 3, 3, 3, 16], [0, 1, 5, 3, 3, 3, 16]]
    )
    np.testing.assert_array_equal(
        batch["decoder_target"],
        [[1, 4, 3, 3, 3, 16, 0], [1, 5, 3, 3, 3, 16, 0]],
    )

  def test_log_prob_and_gradient_step(self):
    examples = self.model.convert_examples(
        [core.Example(x="hello", y=1.0), core.Example(x="hello", y=1.0)]
    )
    log_probs_before = self.model.log_prob(examples)
    self.assertEqual(log_probs_before.shape, (2,))
    self.assertAlmostEqual(log_probs_before[0].squeeze().item(), -26.81, 1)

    # Update the model. Logprob should improve.
    pytorch_model._train_step(self.model, self.optimizer, examples)
    log_probs_after = self.model.log_prob(examples)
    self.assertAlmostEqual(log_probs_after[0].squeeze().item(), -23.86, 1)

  def test_decode(self):
    examples = self.model.convert_examples([core.Example(x="hello", y=2.123)])
    decoded_ids, output_floats = self.model.decode(examples, num_samples=1024)
    # 1 example, 1024 samples, 6 tokens per sample.
    self.assertEqual(tuple(decoded_ids.shape), (1, 1024, 6))

    self.assertAlmostEqual(output_floats[0, 0], -0.2398)
    self.assertAlmostEqual(output_floats[0, 1], -230200000.0)

    self.assertAlmostEqual(np.median(output_floats), -0.0003233)

    # After updating, the median should get closer to target y.
    pytorch_model._train_step(self.model, self.optimizer, examples)
    _, output_floats = self.model.decode(examples, num_samples=128)
    self.assertAlmostEqual(np.median(output_floats), 2.233)


if __name__ == "__main__":
  absltest.main()
