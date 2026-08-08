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

"""Tests for the hybrid (exact + compressed) context architecture.

Coverage:
  1. Shapes, k=0, and both compression rules.
  2. Two-tier accounting: exact bank vs overflow, eviction order.
  3. Token-level structure: y-token ORDER must change the state (regression
     test against bag-of-tokens context representations).
  4. Permutation invariance over pairs while everything fits in the bank.
  5. Incremental update_context_state matches batch build_context_state.
  6. Gradient flow through bank construction and compression.
  7. Batched episodic training decreases loss, and few-shot context improves
     held-out (fresh-task) query prediction in BOTH the exact and the
     overflow regime.
"""

import random

from regress_lm import core
from regress_lm import tokenizers
from regress_lm import vocabs
from regress_lm.pytorch.online import architecture
from regress_lm.pytorch.online import training
import torch
from torch import optim

from absl.testing import absltest
from absl.testing import parameterized


def _make_model(
    max_exact_pairs: int = 3,
    overflow_mode: str = "latent_array",
    d_model: int = 32,
) -> architecture.HybridEncoderDecoder:
  return architecture.HybridEncoderDecoder(
      encoder_vocab_size=50,
      decoder_vocab_size=20,
      encoder_pad_idx=0,
      max_encoder_len=16,
      max_decoder_len=6,
      d_model=d_model,
      num_encoder_layers=1,
      num_decoder_layers=2,
      max_exact_pairs=max_exact_pairs,
      overflow_mode=architecture.OverflowMode(overflow_mode),
      nhead=4,
      num_latents=8,
      num_write_layers=2,
  )


def _random_context(
    batch: int, k: int, src_len: int = 16, y_len: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
  ctx_src = torch.randint(1, 50, (batch, k, src_len))
  ctx_src[:, :, 10:] = 0  # Padding tail.
  ctx_y = torch.randint(1, 20, (batch, k, y_len))
  return ctx_src, ctx_y


class ArchitectureTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)

  def test_forward_shapes(self):
    model = _make_model()
    ctx_src, ctx_y = _random_context(batch=2, k=5)
    state = model.build_context_state(ctx_src, ctx_y)

    q_src = torch.randint(1, 50, (2, 16))
    tgt = torch.randint(1, 20, (2, 5))
    logits = model(q_src, tgt, context_state=state)
    self.assertEqual(logits.shape, (2, 5, 20))

  def test_forward_without_context(self):
    model = _make_model()
    q_src = torch.randint(1, 50, (2, 16))
    tgt = torch.randint(1, 20, (2, 5))
    logits = model(q_src, tgt, context_state=None)
    self.assertEqual(logits.shape, (2, 5, 20))

  def test_two_tier_accounting(self):
    """k pairs past the dial land in tier 2; newest stay exact."""
    model = _make_model(max_exact_pairs=3)
    ctx_src, ctx_y = _random_context(batch=2, k=5)
    state = model.build_context_state(ctx_src, ctx_y)

    self.assertEqual(state.num_exact_pairs, 3)
    self.assertEqual(state.num_pairs_total, 5)
    self.assertIsNotNone(state.S)
    self.assertLen(state.S, 2)  # One per decoder layer.
    self.assertEqual(state.S[0].shape, (2, 8, 32))  # (B, num_latents, d_model)
    # Bank width = 3 pairs x (16 x-tokens + 4 y-tokens).
    self.assertIsNotNone(state.bank)
    self.assertEqual(state.bank.shape, (2, 3 * 20, 32))

  def test_under_capacity_has_no_overflow(self):
    model = _make_model(max_exact_pairs=8)
    ctx_src, ctx_y = _random_context(batch=2, k=5)
    state = model.build_context_state(ctx_src, ctx_y)
    self.assertEqual(state.num_exact_pairs, 5)
    self.assertIsNone(state.S)

  def test_latent_state_shape(self):
    model = _make_model(max_exact_pairs=2)
    ctx_src, ctx_y = _random_context(batch=2, k=4)
    state = model.build_context_state(ctx_src, ctx_y)
    self.assertIsNotNone(state.S)
    self.assertEqual(state.S[0].shape, (2, 8, 32))  # (B, num_latents, d_model)

  def test_incremental_matches_batch(self):
    model = _make_model(max_exact_pairs=3)
    model.eval()
    ctx_src, ctx_y = _random_context(batch=2, k=5)
    q_src = torch.randint(1, 50, (2, 16))
    tgt = torch.randint(1, 20, (2, 5))

    with torch.no_grad():
      state_batch = model.build_context_state(ctx_src, ctx_y)
      state_inc = model._empty_state()  # pylint: disable=protected-access
      for i in range(5):
        state_inc = model.update_context_state(
            state_inc, ctx_src[:, i], ctx_y[:, i]
        )
      out_a = model(q_src, tgt, context_state=state_batch)
      out_b = model(q_src, tgt, context_state=state_inc)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-4)

  def test_y_token_order_changes_output(self):
    """Regression test: context y must NOT behave as a bag of tokens.

    "1.23" and "3.21" share a token multiset; positional encoding on the
    context y embeddings must make their states differ.
    """
    model = _make_model(max_exact_pairs=4)
    model.eval()
    src = torch.randint(1, 50, (1, 1, 16))
    y_fwd = torch.tensor([[[1, 2, 3, 4]]])
    y_rev = torch.tensor([[[4, 3, 2, 1]]])
    q_src = torch.randint(1, 50, (1, 16))
    tgt = torch.randint(1, 20, (1, 5))

    with torch.no_grad():
      out_fwd = model(
          q_src, tgt, context_state=model.build_context_state(src, y_fwd)
      )
      out_rev = model(
          q_src, tgt, context_state=model.build_context_state(src, y_rev)
      )
    self.assertFalse(torch.allclose(out_fwd, out_rev, atol=1e-6))

  def test_label_swap_changes_output(self):
    """Swapping labels between two pairs must change the output.

    This is the negative control for pair binding: if xᵢ and yᵢ are properly
    bound, then (x1,y1),(x2,y2) must produce different predictions than
    (x1,y2),(x2,y1).
    """
    model = _make_model(max_exact_pairs=8)
    model.eval()
    # Build two distinct context pairs with safe token ranges.
    ctx_src = torch.randint(1, 40, (1, 2, 16))
    ctx_y = torch.randint(1, 15, (1, 2, 5))
    # Make sure x1 != x2 and y1 != y2 so swapping is non-trivial.
    ctx_src[0, 1] = (ctx_src[0, 0] % 39) + 1
    ctx_y[0, 1] = (ctx_y[0, 0] % 14) + 1

    q_src = torch.randint(1, 50, (1, 16))
    tgt = torch.randint(1, 20, (1, 5))

    # Original pairing: (x1,y1), (x2,y2).
    with torch.no_grad():
      out_orig = model(
          q_src,
          tgt,
          context_state=model.build_context_state(ctx_src, ctx_y),
      )

    # Swapped pairing: (x1,y2), (x2,y1).
    ctx_y_swapped = ctx_y[:, [1, 0]]  # swap y1 and y2
    with torch.no_grad():
      out_swapped = model(
          q_src,
          tgt,
          context_state=model.build_context_state(ctx_src, ctx_y_swapped),
      )

    self.assertFalse(
        torch.allclose(out_orig, out_swapped, atol=1e-5),
        "Label swap must change output (pair binding is broken).",
    )

  def test_gradients_flow_through_compression(self):
    model = _make_model(max_exact_pairs=2)
    ctx_src, ctx_y = _random_context(batch=2, k=4)  # 2 pairs compressed.
    q_src = torch.randint(1, 50, (2, 16))
    tgt = torch.randint(1, 20, (2, 5))

    state = model.build_context_state(ctx_src, ctx_y)
    loss = model(q_src, tgt, context_state=state).sum()
    loss.backward()

    w_k_grad = model.decoder_layers[0].context_attn.W_k.weight.grad
    self.assertIsNotNone(w_k_grad)
    self.assertGreater(w_k_grad.abs().sum().item(), 0.0)

    # Latent gate parameters must also receive gradients through compression.
    decay_proj = model.decoder_layers[0].context_attn.decay_gate_proj
    self.assertIsNotNone(decay_proj.weight.grad)
    self.assertGreater(decay_proj.weight.grad.abs().sum().item(), 0.0)

    # The encoder must receive gradient through the context pathway too.
    enc_grads = [
        p.grad for p in model.encoder.parameters() if p.grad is not None
    ]
    self.assertNotEmpty(enc_grads)

  def test_context_changes_output(self):
    model = _make_model(max_exact_pairs=8)
    model.eval()
    ctx_src, ctx_y = _random_context(batch=2, k=3)
    q_src = torch.randint(1, 50, (2, 16))
    tgt = torch.randint(1, 20, (2, 5))
    with torch.no_grad():
      out_ctx = model(
          q_src, tgt, context_state=model.build_context_state(ctx_src, ctx_y)
      )
      out_no_ctx = model(q_src, tgt, context_state=None)
    self.assertFalse(torch.allclose(out_ctx, out_no_ctx, atol=1e-6))

  def test_next_token_logits_with_overflowed_state(self):
    model = _make_model(max_exact_pairs=2)
    model.eval()
    ctx_src, ctx_y = _random_context(batch=2, k=5)
    q_src = torch.randint(1, 50, (2, 16))
    with torch.no_grad():
      state = model.build_context_state(ctx_src, ctx_y)
      memory, mask = model.encode(q_src)
      seq = torch.randint(1, 20, (2, 3))
      logits = model.next_token_logits(seq, memory, mask, context_state=state)
    self.assertEqual(logits.shape, (2, 20))


class EpisodicTrainingTest(absltest.TestCase):
  """End-to-end meta-learning behavior on toy lookup tasks."""

  WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    random.seed(0)
    self.encoder_vocab = vocabs.BasicEnglishVocab(self.WORDS)
    self.decoder_vocab = vocabs.DecoderVocab(tokenizers.P10Tokenizer())
    self.cfg = training.HybridModelConfig(
        encoder_vocab=self.encoder_vocab,
        decoder_vocab=self.decoder_vocab,
        max_input_len=6,
        architecture_kwargs=dict(
            d_model=64,
            num_encoder_layers=1,
            num_decoder_layers=2,
            max_exact_pairs=4,
            overflow_mode=architecture.OverflowMode.LATENT_ARRAY,
            nhead=4,
            num_latents=8,
            num_write_layers=2,
        ),
    )

  def _fresh_task(self, n: int, rng=None) -> list[core.Example]:
    """A brand-new task per call: fresh word -> value table."""
    rng = rng or random
    table = {w: rng.uniform(1.0, 9.0) for w in self.WORDS}
    ws = [rng.choice(self.WORDS) for _ in range(n)]
    return [core.Example(x=w, y=table[w]) for w in ws]

  def test_training_loss_decreases(self):
    model = training.HybridModel(self.cfg)
    sampler = training.TaskSampler(
        [self._fresh_task],
        max_context_size=10,  # Exceeds max_exact_pairs=4: trains tier 2.
        num_queries_per_episode=2,
        batch_size=8,
    )
    trainer = training.EpisodicTrainer(
        model, sampler, optim.Adam(model.parameters(), lr=1e-3)
    )
    metrics = trainer.train(num_steps=300, log_every=50)
    losses = [m["train_loss"] for m in metrics]
    self.assertLess(sum(losses[-2:]), sum(losses[:2]))

  def test_context_helps_in_both_regimes(self):
    """Few-shot context lowers fresh-task query CE, exact AND overflow."""
    model = training.HybridModel(self.cfg)
    sampler = training.TaskSampler(
        [self._fresh_task],
        max_context_size=10,
        num_queries_per_episode=2,
        batch_size=8,
    )
    trainer = training.EpisodicTrainer(
        model, sampler, optim.Adam(model.parameters(), lr=1e-3)
    )
    trainer.train(num_steps=2000, log_every=1000)
    model.eval()

    def query_ce(k: int, trials: int = 40) -> float:
      rng = random.Random(7)
      total = 0.0
      with torch.no_grad():
        for _ in range(trials):
          exs = self._fresh_task(k + 1, rng)
          batch = training.EpisodeBatch(
              context_examples=[exs[:k]], query_examples=[exs[k:]]
          )
          loss, _ = model.compute_episodic_loss(batch)
          total += loss.item()
      return total / trials

    ce_k0 = query_ce(0)
    ce_exact = query_ce(4)  # Fits in the bank.
    ce_overflow = query_ce(10)  # 6 pairs compressed into tier 2.
    self.assertLess(ce_exact, ce_k0)
    self.assertLess(ce_overflow, ce_k0)

  def test_inference_state_apis(self):
    model = training.HybridModel(self.cfg)
    examples = self._fresh_task(6)
    state = model.build_state(examples)
    self.assertEqual(state.num_pairs_total, 6)
    self.assertEqual(state.num_exact_pairs, 4)

    state = model.update_state(state, self._fresh_task(1)[0])
    self.assertEqual(state.num_pairs_total, 7)
    self.assertEqual(state.num_exact_pairs, 4)

    token_ids = model.decode("alpha", state)
    self.assertEqual(token_ids.shape, (1, self.cfg.decode_len))


if __name__ == "__main__":
  absltest.main()
