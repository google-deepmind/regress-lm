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
from absl.testing import parameterized

IS_G3 = False


class ModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.encoder_vocab = vocabs.BasicEnglishVocab(['hello', 'world'])
    self.decoder_vocab = vocabs.DecoderVocab(tokenizers.P10Tokenizer())
    self.architecture_kwargs = dict(
        d_model=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    self.model = pytorch_model.PyTorchModel(
        encoder_vocab=self.encoder_vocab,
        decoder_vocab=self.decoder_vocab,
        max_input_len=4,
        compile_model=not IS_G3,
        **self.architecture_kwargs,
    )
    self.optimizer = optim.Adafactor(
        filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.1
    )

  def test_convert(self):
    batch = self.model.convert_examples(
        [core.Example(x='hello', y=1.0), core.Example(x='world', y=2.0)]
    )

    np.testing.assert_array_equal(
        batch['encoder_input'], [[2, 0, 0, 0], [3, 0, 0, 0]]
    )
    np.testing.assert_array_equal(
        batch['decoder_input'], [[0, 1, 4, 3, 3, 3, 16], [0, 1, 5, 3, 3, 3, 16]]
    )
    np.testing.assert_array_equal(
        batch['decoder_target'],
        [[1, 4, 3, 3, 3, 16, 0], [1, 5, 3, 3, 3, 16, 0]],
    )

  def test_log_prob_and_gradient_step(self):
    examples = self.model.convert_examples(
        [core.Example(x='hello', y=1.0), core.Example(x='hello', y=1.0)]
    )
    log_probs_before = self.model.log_prob(examples)
    self.assertEqual(log_probs_before.shape, (2,))
    self.assertAlmostEqual(log_probs_before[0].squeeze().item(), -26.93, 1)

    # Update the model. Logprob should improve.
    pytorch_model._train_step(self.model, self.optimizer, examples)
    log_probs_after = self.model.log_prob(examples)
    self.assertAlmostEqual(log_probs_after[0].squeeze().item(), -18.11, 1)

  def test_decode(self):
    batch = self.model.convert_examples([core.Example(x='hello', y=2.123)])
    decoded_ids, output_floats = self.model.decode(batch, num_samples=1024)
    # 1 example, 1024 samples, 6 tokens per objective, 1 objective total.
    self.assertEqual(tuple(decoded_ids.shape), (1, 1024, 6))

    # 1 example, 1024 samples, 1 objective total.
    self.assertEqual(tuple(output_floats.shape), (1, 1024, 1))

    self.assertAlmostEqual(output_floats[0, 0], -0.2398)
    self.assertAlmostEqual(output_floats[0, 1], -230200000.0)

    self.assertAlmostEqual(np.median(output_floats), -0.00031275)

    # After updating, the median should get closer to target y.
    pytorch_model._train_step(self.model, self.optimizer, batch)
    _, output_floats = self.model.decode(batch, num_samples=128)
    self.assertAlmostEqual(np.median(output_floats), 2.331)

  def test_decode_special_tokens(self):
    tokenizer = tokenizers.AddSpecialValues(tokenizers.P10Tokenizer())
    model = pytorch_model.PyTorchModel(
        encoder_vocab=self.encoder_vocab,
        decoder_vocab=vocabs.DecoderVocab(tokenizer),
        max_input_len=4,
        compile_model=not IS_G3,
        **self.architecture_kwargs,
    )
    batch = model.convert_examples(
        [core.Example(x='hello', y=float('-inf')), core.Example(x='bye', y=1.0)]
    )
    decoded_ids, _ = model.decode(batch, num_samples=5)

    np.testing.assert_array_equal(decoded_ids[0, 0], [36, 36, 36, 36, 36, 36])
    np.testing.assert_array_equal(decoded_ids[0, 1], [1, 4, 9, 4, 7, 33])

  def test_multiobjective(self):
    model = pytorch_model.PyTorchModel(
        encoder_vocab=self.encoder_vocab,
        decoder_vocab=self.decoder_vocab,
        max_input_len=4,
        max_num_objs=2,
        compile_model=not IS_G3,
        **self.architecture_kwargs,
    )
    self.assertEqual(model.decode_len, 12)

    examples = [core.ExampleInput(x='hello'), core.ExampleInput(x='world')]
    batch = model.convert_inputs(examples)

    decoded_ids, output_floats = model.decode(batch, num_samples=1024)
    # Now 12 = 2 objectives * 6 tokens per objective.
    self.assertEqual(tuple(decoded_ids.shape), (2, 1024, 12))
    # Now 2 objectives for last axis.
    self.assertEqual(tuple(output_floats.shape), (2, 1024, 2))


if __name__ == '__main__':
  absltest.main()
