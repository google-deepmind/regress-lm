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

import copy
import dataclasses
import functools
from typing import cast
import numpy as np
from regress_lm import core
from regress_lm import tokenizers
from regress_lm import vocabs
from regress_lm.pytorch import fine_tuning
from regress_lm.pytorch import model as model_lib
import torch
from torch import optim
from absl.testing import absltest
from absl.testing import parameterized


class ModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.encoder_vocab = vocabs.BasicEnglishVocab(['hello', 'world'])
    self.decoder_tokenizer = tokenizers.P10Tokenizer()
    self.decoder_vocab = vocabs.DecoderVocab(self.decoder_tokenizer)
    self.architecture_kwargs = dict(
        d_model=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    self.cfg = model_lib.PyTorchModelConfig(
        encoder_vocab=self.encoder_vocab,
        decoder_vocab=self.decoder_vocab,
        max_input_len=4,
        architecture_kwargs=self.architecture_kwargs,
    )
    self.model = self.cfg.make_model(compile_model=False)

    self.fine_tuner = fine_tuning.PyTorchFineTuner(
        self.model,
        optimizer_factory=lambda params: optim.Adafactor(
            filter(lambda p: p.requires_grad, params), lr=0.1
        ),
        max_epochs=1,
    )

  def test_convert(self):
    batch = self.model.converter.convert_examples(
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
    raw_examples = [
        core.Example(x='hello', y=1.0),
        core.Example(x='hello', y=1.0),
    ]
    examples_tensors = self.model.converter.convert_examples(raw_examples)
    log_probs_before = self.model.log_prob(examples_tensors)
    self.assertEqual(log_probs_before.shape, (2,))
    self.assertAlmostEqual(log_probs_before[0].squeeze().item(), -22.69, 1)

    # Update the model. Logprob should improve.
    self.fine_tuner.fine_tune(raw_examples)

    log_probs_after = self.model.log_prob(examples_tensors)
    self.assertAlmostEqual(log_probs_after[0].squeeze().item(), -14.13, 1)

  def test_compute_losses_and_metrics(self):
    """Verifies the exact output of the loss calculation."""
    raw_examples = [
        core.Example(x='hello', y=1.0),
        core.Example(x='world', y=10000.0),
    ]
    examples_tensors = self.model.converter.convert_examples(raw_examples)

    self.model.eval()
    with torch.no_grad():
      losses_per_example, metrics = self.model.compute_losses_and_metrics(
          examples_tensors
      )

    # Check the per-example loss (mean over tokens)
    self.assertEqual(losses_per_example.shape, (2,))
    self.assertAlmostEqual(losses_per_example[0].item(), 3.7821, 3)
    self.assertAlmostEqual(losses_per_example[1].item(), 4.0466, 3)

    # Check the overall batch loss metric
    self.assertAlmostEqual(metrics['loss_mean'].item(), 3.9143, 3)

  def test_decode(self):
    raw_example = [core.Example(x='hello', y=2.123)]
    batch = self.model.converter.convert_examples(raw_example)
    decoded_ids, output_floats = self.model.decode(batch, num_samples=1024)
    # 1 example, 1024 samples, 6 tokens per objective, 1 objective total.
    self.assertEqual(tuple(decoded_ids.shape), (1, 1024, 6))

    # 1 example, 1024 samples, 1 objective total.
    self.assertEqual(tuple(output_floats.shape), (1, 1024, 1))

    self.assertAlmostEqual(output_floats[0, 0, 0], 0.6898)
    self.assertAlmostEqual(output_floats[0, 1, 0], -230800000.0)

    self.assertAlmostEqual(np.median(output_floats), 0.00082365)

    # After updating, the median should get closer to target y.
    self.fine_tuner.fine_tune(raw_example, seed=42)

    _, output_floats = self.model.decode(batch, num_samples=128)
    self.assertAlmostEqual(np.median(output_floats), 2.5275)

  def test_decode_special_tokens(self):
    tokenizer = tokenizers.AddSpecialValues(tokenizers.P10Tokenizer())
    cfg = dataclasses.replace(
        self.cfg,
        decoder_vocab=vocabs.DecoderVocab(tokenizer),
        max_input_len=4,
    )
    model = cfg.make_model(compile_model=False)
    examples = [
        core.Example(x='hello', y=float('-inf')),
        core.Example(x='bye', y=float('-inf')),
    ]

    # Update the model. More likely to decode special tokens.
    self.fine_tuner.fine_tune(examples, seed=42)
    batch = model.converter.convert_examples(examples)
    decoded_ids, _ = model.decode(batch, num_samples=5)

    np.testing.assert_array_equal(decoded_ids[0, 0], [37, 37, 37, 37, 37, 37])
    np.testing.assert_array_equal(decoded_ids[0, 1], [2, 5, 9, 5, 9, 31])

  @parameterized.parameters(True, False)
  def test_multiobjective(self, add_separators: bool):
    tokenizer = self.decoder_tokenizer
    if add_separators:
      tokenizer = tokenizers.AppendPadTokenizer(tokenizer)

    cfg = dataclasses.replace(
        self.cfg,
        decoder_vocab=vocabs.DecoderVocab(tokenizer),
        max_input_len=4,
        max_num_objs=2,
    )
    model = cfg.make_model(compile_model=False)

    examples = [core.ExampleInput(x='hello'), core.ExampleInput(x='world')]
    batch = model.converter.convert_inputs(examples)

    decoded_ids, output_floats = model.decode(batch, num_samples=1024)
    if not add_separators:  # 12 = 2 objectives * 6 tokens per objective.
      self.assertEqual(tuple(decoded_ids.shape), (2, 1024, 12))
      self.assertEqual(cfg.decode_len, 12)
    else:  # +2 for padding/separators.
      self.assertEqual(tuple(decoded_ids.shape), (2, 1024, 14))
      self.assertEqual(cfg.decode_len, 14)

    # Now 2 objectives for last axis.
    self.assertEqual(tuple(output_floats.shape), (2, 1024, 2))

  def test_gradient_accumulation_equivalence(self):
    torch.manual_seed(123)

    model_microbatch = copy.deepcopy(self.model)
    model_microbatch.load_state_dict(self.model.state_dict())

    fine_tuner_base = fine_tuning.PyTorchFineTuner(
        model=self.model,
        optimizer_factory=functools.partial(optim.Adafactor, lr=1e-4),
        max_epochs=1,
    )
    fine_tuner_microbatch = fine_tuning.PyTorchFineTuner(
        model=model_microbatch,
        optimizer_factory=functools.partial(optim.Adafactor, lr=1e-4),
        max_epochs=1,
        batch_size_per_device=2,
    )

    train_examples = [core.Example(x='hello', y=1.0)] * 4

    # --- Run both fine-tuning processes ---
    fine_tuner_base.fine_tune(train_examples, seed=42)
    fine_tuner_microbatch.fine_tune(train_examples, seed=42)

    # --- Verification ---
    for p1, p2 in zip(self.model.parameters(), model_microbatch.parameters()):
      torch.testing.assert_close(p1, p2, atol=1e-7, rtol=1e-5)

  def test_early_stopping_with_patience(self):

    class MockFineTuner(fine_tuning.PyTorchFineTuner):
      """Mock fine-tuner that allows us to control the validation losses."""

      def __init__(self, validation_losses, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_losses = validation_losses
        self._val_loss_iter = iter(self.validation_losses)
        self.training_epochs_run = 0
        self.best_state_at_min_loss = None

      def _run_training_epoch(self, *args, **kwargs):
        self.training_epochs_run += 1

      def _run_validation_epoch(self, *args, **kwargs) -> float:
        loss = next(self._val_loss_iter)
        if loss == min(self.validation_losses):
          self.best_state_at_min_loss = copy.deepcopy(
              self.target_model.state_dict()
          )
        return loss

    # Scenario: Loss improves, then gets worse for 3 epochs.
    # The best loss is 5.0, which occurs after the 2nd training epoch.
    validation_losses = [10.0, 8.0, 5.0, 6.0, 7.0, 8.0]

    # We expect training to run for 4 epochs:
    # - Epoch 1 (loss 8.0)
    # - Epoch 2 (loss 5.0) -> Best loss, patience counter resets.
    # - Epoch 3 (loss 6.0) -> No improvement, patience counter is 1.
    # - Epoch 4 (loss 7.0) -> No improvement, patience counter is 2.
    # `should_stop()` is now true, so the loop breaks before epoch 5.
    expected_epochs_run = 4

    mock_tuner = MockFineTuner(
        validation_losses=validation_losses,
        model=self.model,
        optimizer_factory=lambda params: optim.Adafactor(
            filter(lambda p: p.requires_grad, params), lr=0.1
        ),
        max_epochs=10,  # High max_epochs to ensure early stopping is the cause.
        patience=2,
    )

    # Example not used, but we need to pass something.
    mock_tuner.fine_tune(examples=[core.Example(x='hello', y=1.0)])

    # Training stops at right run.
    self.assertEqual(mock_tuner.training_epochs_run, expected_epochs_run)

    # Model state should be the best one that was found at epoch 2.
    final_state = self.model.state_dict()
    best_state = mock_tuner.best_state_at_min_loss
    self.assertIsNotNone(best_state)
    best_state = cast(dict[str, torch.Tensor], best_state)
    for key in best_state:
      torch.testing.assert_close(final_state[key], best_state[key])

  def test_lora_fine_tuning(self):
    # Adafactor is too conservative when using LoRA, use AdamW instead.
    lora_fine_tuner = fine_tuning.PyTorchFineTuner(
        model=self.model,
        optimizer_factory=functools.partial(optim.AdamW, lr=0.1),
        max_epochs=5,  # Need more steps to move model.
        patience=None,
        use_lora=True,
    )

    raw_examples = [
        core.Example(x='hello', y=1.0),
        core.Example(x='hello', y=1.0),
    ]
    examples_tensors = self.model.converter.convert_examples(raw_examples)
    log_probs_before = self.model.log_prob(examples_tensors)
    self.assertEqual(log_probs_before.shape, (2,))
    self.assertAlmostEqual(log_probs_before[0].squeeze().item(), -22.69, 1)

    # Update the model. Logprob should improve.
    lora_fine_tuner.fine_tune(raw_examples)

    log_probs_after = self.model.log_prob(examples_tensors)
    self.assertAlmostEqual(log_probs_after[0].squeeze().item(), -19.24, 1)

    # Allow further fine-tuning.
    lora_fine_tuner.fine_tune(raw_examples)


if __name__ == '__main__':
  absltest.main()
