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

"""Tests for the online (incremental) architecture and training pipeline.

Tests cover:
  1. Architecture shapes and forward passes (with and without context).
  2. Context state construction and incremental updates.
  3. Autoregressive decode loop with context state.
  4. Episodic training loop on a toy regression task.
"""

from regress_lm import core
from regress_lm import tokenizers
from regress_lm import vocabs
from regress_lm.pytorch.online import architecture
from regress_lm.pytorch.online import training
import torch
from torch import optim
from absl.testing import absltest


def _make_model(d_model: int = 16) -> architecture.IncrementalEncoderDecoder:
  """Creates a tiny IncrementalEncoderDecoder for testing."""
  return architecture.IncrementalEncoderDecoder(
      encoder_vocab_size=10,
      decoder_vocab_size=20,
      encoder_pad_idx=0,
      max_encoder_len=15,
      max_decoder_len=7,
      d_model=d_model,
      num_encoder_layers=1,
      num_decoder_layers=2,
  )


class ArchitectureTest(absltest.TestCase):
  """Tests for IncrementalEncoderDecoder architecture."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.model = _make_model()
    self.batch_size = 2
    self.src_seq_len = 10

  def test_forward_without_context(self):
    """Forward pass with context_state=None should work (k=0)."""
    src = torch.randint(1, 10, (self.batch_size, self.src_seq_len))
    tgt = torch.randint(1, 20, (self.batch_size, 7))
    logits = self.model.forward(src, tgt, context_state=None)
    self.assertEqual(logits.shape, (self.batch_size, 7, 20))

  def test_forward_with_context(self):
    """Forward pass with a context state should produce valid logits."""
    # Build context state from 2 pairs.
    ctx_src = [
        torch.randint(1, 10, (self.batch_size, 8)),
        torch.randint(1, 10, (self.batch_size, 6)),
    ]
    ctx_y = [
        torch.randint(1, 20, (self.batch_size, 5)),
        torch.randint(1, 20, (self.batch_size, 5)),
    ]
    state = self.model.build_context_state(ctx_src, ctx_y)

    # Check state shapes.
    num_layers = 2
    self.assertLen(state.S, num_layers)
    self.assertLen(state.z, num_layers)
    # S: (B, H=8, head_dim, head_dim), head_dim = 16//8 = 2
    self.assertEqual(state.S[0].shape, (self.batch_size, 8, 2, 2))
    self.assertEqual(state.z[0].shape, (self.batch_size, 8, 2))

    # Forward with context.
    src = torch.randint(1, 10, (self.batch_size, self.src_seq_len))
    tgt = torch.randint(1, 20, (self.batch_size, 7))
    logits = self.model.forward(src, tgt, context_state=state)
    self.assertEqual(logits.shape, (self.batch_size, 7, 20))

  def test_encode(self):
    """Encode should return memory and padding mask."""
    src = torch.randint(1, 10, (self.batch_size, self.src_seq_len))
    src[0, -2:] = 0  # Add padding.
    memory, mask = self.model.encode(src)
    self.assertEqual(memory.shape, (self.batch_size, self.src_seq_len, 16))
    self.assertEqual(mask.shape, (self.batch_size, self.src_seq_len))
    # Padding positions should be True.
    self.assertTrue(torch.all(mask[0, -2:]).item())
    self.assertFalse(torch.any(mask[0, :-2]).item())

  def test_update_context_state(self):
    """Incremental update should produce same state as building from scratch."""
    torch.manual_seed(0)
    ctx_src_1 = torch.randint(1, 10, (1, 8))
    ctx_y_1 = torch.randint(1, 20, (1, 5))
    ctx_src_2 = torch.randint(1, 10, (1, 6))
    ctx_y_2 = torch.randint(1, 20, (1, 5))

    # Build from scratch with both pairs.
    state_both = self.model.build_context_state(
        [ctx_src_1, ctx_src_2], [ctx_y_1, ctx_y_2]
    )
    # Build incrementally: first pair, then add second.
    state_one = self.model.build_context_state([ctx_src_1], [ctx_y_1])
    state_incremental = self.model.update_context_state(
        state_one, ctx_src_2, ctx_y_2
    )

    # Should be numerically identical.
    for l in range(len(state_both.S)):
      torch.testing.assert_close(
          state_both.S[l], state_incremental.S[l], atol=1e-5, rtol=1e-5
      )
      torch.testing.assert_close(
          state_both.z[l], state_incremental.z[l], atol=1e-5, rtol=1e-5
      )

  def test_next_token_logits_with_context(self):
    """Autoregressive decoding step should work with context state."""
    ctx_src = [torch.randint(1, 10, (1, 8))]
    ctx_y = [torch.randint(1, 20, (1, 5))]
    state = self.model.build_context_state(ctx_src, ctx_y)

    src = torch.randint(1, 10, (1, self.src_seq_len))
    memory, mask = self.model.encode(src)

    # Start with pad token.
    current_tgt = torch.zeros(1, 1, dtype=torch.long)
    logits = self.model.next_token_logits(current_tgt, memory, mask, state)
    self.assertEqual(logits.shape, (1, 20))

    # Extend by one token.
    next_id = torch.argmax(logits, dim=-1, keepdim=True)
    current_tgt = torch.cat([current_tgt, next_id], dim=1)
    logits2 = self.model.next_token_logits(current_tgt, memory, mask, state)
    self.assertEqual(logits2.shape, (1, 20))

  def test_context_changes_output(self):
    """Adding context should change the decoder output (not be ignored)."""
    torch.manual_seed(42)
    src = torch.randint(1, 10, (1, 8))
    tgt = torch.randint(1, 20, (1, 7))

    # Forward without context.
    logits_no_ctx = self.model.forward(src, tgt, context_state=None)

    # Forward with context.
    ctx_src = [torch.randint(1, 10, (1, 8))]
    ctx_y = [torch.randint(1, 20, (1, 5))]
    state = self.model.build_context_state(ctx_src, ctx_y)
    logits_with_ctx = self.model.forward(src, tgt, context_state=state)

    # Outputs should differ.
    self.assertFalse(torch.allclose(logits_no_ctx, logits_with_ctx))


class EpisodicTrainingTest(absltest.TestCase):
  """Tests for the episodic training pipeline."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.encoder_vocab = vocabs.BasicEnglishVocab(['hello', 'world', 'foo'])
    self.decoder_tokenizer = tokenizers.P10Tokenizer()
    self.decoder_vocab = vocabs.DecoderVocab(self.decoder_tokenizer)

  def _make_incremental_model(self) -> training.IncrementalModel:
    cfg = training.IncrementalModelConfig(
        encoder_vocab=self.encoder_vocab,
        decoder_vocab=self.decoder_vocab,
        max_input_len=4,
        architecture_kwargs=dict(
            d_model=16,
            num_encoder_layers=1,
            num_decoder_layers=1,
        ),
    )
    return training.IncrementalModel(cfg)

  def test_episodic_loss_k0(self):
    """Loss computation with k=0 context (no context pairs)."""
    model = self._make_incremental_model()
    episode = training.Episode(
        context_examples=[],
        query_examples=[core.Example(x='hello', y=1.0)],
    )
    loss, metrics = model.compute_episodic_loss(episode)
    self.assertEqual(loss.shape, ())
    self.assertGreater(loss.item(), 0.0)
    self.assertIn('loss_mean', metrics)

  def test_episodic_loss_k2(self):
    """Loss computation with k=2 context pairs."""
    model = self._make_incremental_model()
    episode = training.Episode(
        context_examples=[
            core.Example(x='hello', y=1.0),
            core.Example(x='world', y=2.0),
        ],
        query_examples=[core.Example(x='foo', y=3.0)],
    )
    loss, _ = model.compute_episodic_loss(episode)
    self.assertEqual(loss.shape, ())
    self.assertGreater(loss.item(), 0.0)

  def test_episodic_loss_backward(self):
    """Gradients should flow through the context state construction."""
    model = self._make_incremental_model()
    episode = training.Episode(
        context_examples=[core.Example(x='hello', y=1.0)],
        query_examples=[core.Example(x='world', y=2.0)],
    )
    loss, _ = model.compute_episodic_loss(episode)
    loss.backward()

    # Check that gradients flowed into the context cross-attention params.
    ctx_attn = model.encoder_decoder.decoder_layers[0].context_cross_attn
    self.assertIsNotNone(ctx_attn.W_k_ctx.weight.grad)
    self.assertFalse(torch.all(ctx_attn.W_k_ctx.weight.grad == 0).item())

  def test_training_loop_loss_decreases(self):
    """A few training steps should decrease loss on a constant task."""
    model = self._make_incremental_model()

    # Simple task: always return y=1.0 regardless of x.
    def constant_task(n: int) -> list[core.Example]:
      return [core.Example(x='hello', y=1.0) for _ in range(n)]

    sampler = training.TaskSampler(
        tasks=[constant_task],
        max_context_size=2,
        num_queries_per_episode=1,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = training.EpisodicTrainer(model, sampler, optimizer)

    # Collect losses over a few steps.
    losses = []
    for _ in range(20):
      step_metrics = trainer.train_step()
      losses.append(step_metrics['train_loss'])

    # Loss should generally decrease (compare first 5 avg to last 5 avg).
    early_avg = sum(losses[:5]) / 5
    late_avg = sum(losses[-5:]) / 5
    self.assertLess(late_avg, early_avg)

  def test_build_and_update_state_inference(self):
    """Inference-time build_state and update_state should work."""
    model = self._make_incremental_model()

    examples = [
        core.Example(x='hello', y=1.0),
        core.Example(x='world', y=2.0),
    ]
    state = model.build_state(examples)
    self.assertIsInstance(state, architecture.ContextState)

    # Incrementally add one more.
    new_state = model.update_state(state, core.Example(x='foo', y=3.0))
    self.assertIsInstance(new_state, architecture.ContextState)

    # S should have changed.
    self.assertFalse(torch.allclose(state.S[0], new_state.S[0]))


class BehavioralTest(absltest.TestCase):
  """Sanity-checks that the model learns to use context examples.

  The core value proposition: after episodic training, providing context
  (x, y) pairs from the same task should improve the model's predictions
  on a query from that task, compared to predicting with no context (k=0).

  We use the simplest possible task family: constant functions.
  Task_c: for any input x, the output is always y = c.
  Different tasks have different c values.
  """

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    self.words = [
        'hello',
        'world',
        'foo',
        'bar',
        'test',
        'run',
        'data',
        'val',
    ]
    self.encoder_vocab = vocabs.BasicEnglishVocab(self.words)
    self.decoder_tokenizer = tokenizers.P10Tokenizer()
    self.decoder_vocab = vocabs.DecoderVocab(self.decoder_tokenizer)

  def _make_model(self) -> training.IncrementalModel:
    cfg = training.IncrementalModelConfig(
        encoder_vocab=self.encoder_vocab,
        decoder_vocab=self.decoder_vocab,
        max_input_len=4,
        architecture_kwargs=dict(
            d_model=32,
            num_encoder_layers=1,
            num_decoder_layers=2,
        ),
    )
    return training.IncrementalModel(cfg)

  def _make_constant_task(self, c: float):
    """Returns a task callable that always outputs y=c."""
    words = self.words

    def task(n: int) -> list[core.Example]:
      import random  # pylint: disable=g-import-not-at-top

      return [core.Example(x=random.choice(words), y=c) for _ in range(n)]

    return task

  def _eval_loss(
      self,
      model: training.IncrementalModel,
      query: core.Example,
      context: list[core.Example],
  ) -> float:
    """Evaluates cross-entropy loss for a single query with optional context."""
    model.eval()
    episode = training.Episode(
        context_examples=context,
        query_examples=[query],
    )
    with torch.no_grad():
      loss, _ = model.compute_episodic_loss(episode)
    return loss.item()

  def test_context_improves_prediction_after_training(self):
    """After training, k>0 context should yield lower loss than k=0.

    Protocol:
      1. Create tasks: y=1.0, y=2.0, y=3.0 (training).
      2. Train episodically for N steps.
      3. Evaluate on a HELD-OUT task y=2.0 (same distribution but fresh query):
         - k=0: no context.
         - k=3: three examples from the same task as context.
      4. Assert loss(k=3) < loss(k=0).
    """
    import random  # pylint: disable=g-import-not-at-top

    random.seed(0)

    model = self._make_model()

    # Training tasks with different constant values.
    train_tasks = [
        self._make_constant_task(c) for c in [1.0, 2.0, 3.0, 4.0, 5.0]
    ]

    sampler = training.TaskSampler(
        tasks=train_tasks,
        max_context_size=3,
        num_queries_per_episode=1,
    )
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    trainer = training.EpisodicTrainer(model, sampler, optimizer)

    # Train for enough steps to learn the "copy from context" behavior.
    for _ in range(200):
      trainer.train_step()

    # Evaluate on a held-out query.
    eval_query = core.Example(x='data', y=2.0)

    # k=0: no context — model must guess blindly.
    loss_no_context = self._eval_loss(model, eval_query, context=[])

    # k=3: three context examples from the same task.
    context_examples = [
        core.Example(x='hello', y=2.0),
        core.Example(x='world', y=2.0),
        core.Example(x='foo', y=2.0),
    ]
    loss_with_context = self._eval_loss(
        model, eval_query, context=context_examples
    )

    # The core assertion: context should help.
    self.assertLess(
        loss_with_context,
        loss_no_context,
        'Context should reduce loss. '
        f'Got loss_k0={loss_no_context:.4f}, loss_k3={loss_with_context:.4f}',
    )

  def test_more_context_helps_more(self):
    """Loss should decrease monotonically as more context is added.

    After training, k=3 should be better than k=1 which should be
    better than k=0.
    """
    import random  # pylint: disable=g-import-not-at-top

    random.seed(0)

    model = self._make_model()

    train_tasks = [
        self._make_constant_task(c) for c in [1.0, 3.0, 5.0, 7.0, 9.0]
    ]

    sampler = training.TaskSampler(
        tasks=train_tasks,
        max_context_size=4,
        num_queries_per_episode=1,
    )
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    trainer = training.EpisodicTrainer(model, sampler, optimizer)

    for _ in range(200):
      trainer.train_step()

    eval_query = core.Example(x='val', y=3.0)

    loss_k0 = self._eval_loss(model, eval_query, context=[])
    loss_k1 = self._eval_loss(
        model,
        eval_query,
        context=[core.Example(x='hello', y=3.0)],
    )
    loss_k3 = self._eval_loss(
        model,
        eval_query,
        context=[
            core.Example(x='hello', y=3.0),
            core.Example(x='world', y=3.0),
            core.Example(x='foo', y=3.0),
        ],
    )

    # At minimum, context should help over no context.
    self.assertLess(
        loss_k1,
        loss_k0,
        f'k=1 should beat k=0. Got k0={loss_k0:.4f}, k1={loss_k1:.4f}',
    )
    # More context should help (or at least not hurt).
    self.assertLessEqual(
        loss_k3,
        loss_k1,
        f'k=3 should beat k=1. Got k1={loss_k1:.4f}, k3={loss_k3:.4f}',
    )

  def test_incremental_update_matches_batch(self):
    """Inference via update_state should produce same loss as batch build."""
    import random  # pylint: disable=g-import-not-at-top

    random.seed(42)

    model = self._make_model()

    # Quick training to make the model non-trivial.
    train_tasks = [self._make_constant_task(c) for c in [1.0, 2.0]]
    sampler = training.TaskSampler(
        tasks=train_tasks,
        max_context_size=2,
        num_queries_per_episode=1,
    )
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    trainer = training.EpisodicTrainer(model, sampler, optimizer)
    for _ in range(50):
      trainer.train_step()

    # Build state two ways.
    ex1 = core.Example(x='hello', y=1.0)
    ex2 = core.Example(x='world', y=1.0)
    query = core.Example(x='foo', y=1.0)

    # Way 1: batch build.
    loss_batch = self._eval_loss(model, query, context=[ex1, ex2])

    # Way 2: incremental build.
    state = model.build_state([ex1])
    state = model.update_state(state, ex2)
    # Manually set up context state and compute loss.
    model.eval()
    with torch.no_grad():
      query_src = model._to_device(  # pylint: disable=protected-access
          model._encode_text(query.x).unsqueeze(0)  # pylint: disable=protected-access
      )
      y_tokens = model.cfg.decoder_vocab.to_token_ids(query.y)
      dec_len = model.cfg.decode_len
      pad_id = model.cfg.decoder_vocab.bos_pad_id

      import numpy as np  # pylint: disable=g-import-not-at-top

      dec_input = np.full(dec_len + 1, pad_id, dtype=np.int64)
      dec_target = np.full(dec_len + 1, pad_id, dtype=np.int64)
      insert_len = min(len(y_tokens), dec_len)
      dec_input[1 : insert_len + 1] = y_tokens[:insert_len]
      dec_target[:insert_len] = y_tokens[:insert_len]

      dec_input_t = model._to_device(  # pylint: disable=protected-access
          torch.from_numpy(dec_input).unsqueeze(0)
      )
      dec_target_t = model._to_device(  # pylint: disable=protected-access
          torch.from_numpy(dec_target).unsqueeze(0)
      )

      logits = model.encoder_decoder.forward(
          query_src, dec_input_t, context_state=state
      )
      import torch.nn.functional as F  # pylint: disable=g-import-not-at-top

      loss_incremental = F.cross_entropy(
          logits.squeeze(0),
          dec_target_t.squeeze(0),
          ignore_index=pad_id,
      ).item()

    self.assertAlmostEqual(loss_batch, loss_incremental, places=4)


if __name__ == '__main__':
  absltest.main()
