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

"""Batched episodic pretraining for HybridEncoderDecoder.

Each training step draws a batch of episodes, all sharing one context size k
(sampled per step) so tensors batch cleanly:

  - Each batch element is an independent episode: k context (x, y) pairs and
    Q query pairs drawn from one task.
  - k is sampled from [0, max_context_size], where max_context_size should
    EXCEED the model's max_exact_pairs so the compression pathway (tier 2)
    receives gradient during pretraining.  A model never trained past its
    exact-bank capacity will not have learned to read its compressed state.
  - The loss is teacher-forced cross-entropy on the query y tokens, averaged
    over queries and episodes; gradients flow through bank construction and
    (on overflow) through the compression updates.
"""

import dataclasses
import logging
import random
from typing import Any, Callable, Sequence

import numpy as np
from regress_lm import core
from regress_lm import vocabs
from regress_lm.pytorch.online import architecture
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

Tensor = torch.Tensor
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EpisodeBatch:
  """A batch of episodes sharing one context size k.

  Attributes:
    context_examples: Per-episode context pairs; all lists have equal length k.
    query_examples: Per-episode query pairs; all lists have equal length Q.
  """

  context_examples: list[list[core.Example]]
  query_examples: list[list[core.Example]]

  @property
  def batch_size(self) -> int:
    """Number of episodes in the batch."""
    return len(self.context_examples)

  @property
  def k(self) -> int:
    """Number of context pairs per episode."""
    return len(self.context_examples[0])

  @property
  def num_queries(self) -> int:
    """Number of query pairs per episode."""
    return len(self.query_examples[0])


class TaskSampler:
  """Samples batches of episodes from a collection of tasks."""

  def __init__(
      self,
      tasks: Sequence[Callable[[int], list[core.Example]]],
      max_context_size: int = 128,
      num_queries_per_episode: int = 1,
      batch_size: int = 8,
  ):
    """Initializes the sampler.

    Args:
      tasks: Callables; task(n) returns n Examples from that task.
      max_context_size: Upper bound of the per-step context size k.  Set this
        ABOVE the model's max_exact_pairs to train the compression tier.
      num_queries_per_episode: Q query pairs per episode.
      batch_size: Episodes per batch.
    """
    self.tasks = tasks
    self.max_context_size = max_context_size
    self.num_queries_per_episode = num_queries_per_episode
    self.batch_size = batch_size

  def sample_batch(self) -> EpisodeBatch:
    """Samples one batch; a single k is shared across the batch."""
    k = random.randint(0, self.max_context_size)
    ctx, qry = [], []
    for _ in range(self.batch_size):
      task = random.choice(self.tasks)
      examples = task(k + self.num_queries_per_episode)
      ctx.append(examples[:k])
      qry.append(examples[k:])
    return EpisodeBatch(context_examples=ctx, query_examples=qry)


@dataclasses.dataclass(frozen=True)
class HybridModelConfig:
  """Configuration for a HybridModel.

  Attributes:
    encoder_vocab: Vocabulary for encoding text inputs x.
    decoder_vocab: Vocabulary for decoding float outputs y.
    max_input_len: Maximum encoder input length.
    architecture_kwargs: Kwargs for HybridEncoderDecoder (includes the
      ``max_exact_pairs`` dial and ``overflow_mode``).
    max_num_objs: Number of float objects per example.
  """

  encoder_vocab: vocabs.EncoderVocab[str]
  decoder_vocab: vocabs.DecoderVocab[float]
  max_input_len: int
  architecture_kwargs: dict[str, Any]
  max_num_objs: int = 1

  @property
  def decode_len(self) -> int:
    """The number of tokens expected per decoded sequence."""
    return self.max_num_objs * self.decoder_vocab.num_tokens_per_obj


class HybridModel(nn.Module):
  """Wraps HybridEncoderDecoder with episodic training and inference APIs."""

  def __init__(self, config: HybridModelConfig):
    super().__init__()
    self.cfg = config
    self.encoder_decoder = architecture.HybridEncoderDecoder(
        encoder_vocab_size=len(self.cfg.encoder_vocab),
        decoder_vocab_size=len(self.cfg.decoder_vocab),
        encoder_pad_idx=self.cfg.encoder_vocab.pad_id,
        max_encoder_len=self.cfg.max_input_len,
        max_decoder_len=self.cfg.decode_len + 1,
        **self.cfg.architecture_kwargs,
    )

  @property
  def device(self) -> torch.device:
    return next(self.parameters()).device

  # ---- tensorization ------------------------------------------------------

  def _encode_text_np(self, text: str) -> np.ndarray:
    token_ids = self.cfg.encoder_vocab.to_token_ids(text)
    padded = np.full(
        self.cfg.max_input_len, self.cfg.encoder_vocab.pad_id, dtype=np.int64
    )
    n = min(len(token_ids), self.cfg.max_input_len)
    padded[:n] = token_ids[:n]
    return padded

  def _context_tensors(
      self, context_examples: list[list[core.Example]]
  ) -> tuple[Tensor, Tensor]:
    """Batches context pairs to (B, k, L) src ids and (B, k, T) y ids."""
    src = np.stack([
        np.stack([self._encode_text_np(ex.x) for ex in episode])
        for episode in context_examples
    ])  # (B, k, L)
    y = np.stack([
        np.stack([
            np.asarray(
                self.cfg.decoder_vocab.to_token_ids(ex.y), dtype=np.int64
            )
            for ex in episode
        ])
        for episode in context_examples
    ])  # (B, k, T)
    dev = self.device
    return (
        torch.from_numpy(src).to(dev, non_blocking=True),
        torch.from_numpy(y).to(dev, non_blocking=True),
    )

  def _query_tensors(
      self, query_examples: list[list[core.Example]]
  ) -> tuple[Tensor, Tensor, Tensor]:
    """Flattens (B episodes x Q queries) to (B*Q, ...) tensors.

    Args:
      query_examples: list of per-episode query pairs.

    Returns:
      (query_src, dec_input, dec_target) with shapes
      (B*Q, L), (B*Q, dec_len+1), (B*Q, dec_len+1).
    """
    dec_len = self.cfg.decode_len
    pad_id = self.cfg.decoder_vocab.bos_pad_id
    srcs, dec_inputs, dec_targets = [], [], []
    for episode in query_examples:
      for ex in episode:
        srcs.append(self._encode_text_np(ex.x))
        y_tokens = self.cfg.decoder_vocab.to_token_ids(ex.y)
        dec_input = np.full(dec_len + 1, pad_id, dtype=np.int64)
        dec_target = np.full(dec_len + 1, pad_id, dtype=np.int64)
        n = min(len(y_tokens), dec_len)
        dec_input[1 : n + 1] = y_tokens[:n]
        dec_target[:n] = y_tokens[:n]
        dec_inputs.append(dec_input)
        dec_targets.append(dec_target)
    dev = self.device
    to = lambda a: torch.from_numpy(np.stack(a)).to(dev, non_blocking=True)
    return to(srcs), to(dec_inputs), to(dec_targets)

  # ---- training loss ------------------------------------------------------

  def compute_episodic_loss(
      self, batch: EpisodeBatch
  ) -> tuple[Tensor, dict[str, Tensor]]:
    """Teacher-forced CE over query y tokens, batched over episodes.

    Args:
      batch: An EpisodeBatch (all episodes share k and Q).

    Returns:
      (loss, metrics).
    """
    context_state = None
    if batch.k > 0:
      ctx_src, ctx_y = self._context_tensors(batch.context_examples)
      context_state = self.encoder_decoder.build_context_state(ctx_src, ctx_y)
      if batch.num_queries > 1:
        context_state = context_state.expand_for_queries(batch.num_queries)

    query_src, dec_input, dec_target = self._query_tensors(batch.query_examples)
    logits = self.encoder_decoder(
        query_src, dec_input, context_state=context_state
    )  # (B*Q, dec_len+1, V)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        dec_target.reshape(-1),
        ignore_index=self.cfg.decoder_vocab.bos_pad_id,
    )
    metrics = {
        "loss_mean": loss.detach(),
        "num_exact_pairs": torch.tensor(
            0 if context_state is None else context_state.num_exact_pairs
        ),
        "num_compressed_pairs": torch.tensor(
            0
            if context_state is None
            else context_state.num_pairs_total - context_state.num_exact_pairs
        ),
    }
    return loss, metrics

  # ---- inference APIs -----------------------------------------------------

  @torch.no_grad()
  def build_state(
      self, examples: Sequence[core.Example]
  ) -> architecture.HybridContextState:
    """Builds a context state from user few-shot examples (B=1)."""
    ctx_src, ctx_y = self._context_tensors([list(examples)])
    return self.encoder_decoder.build_context_state(ctx_src, ctx_y).detached()

  @torch.no_grad()
  def update_state(
      self,
      state: architecture.HybridContextState,
      new_example: core.Example,
  ) -> architecture.HybridContextState:
    """Adds one (x, y) pair to an existing state."""
    src = torch.from_numpy(self._encode_text_np(new_example.x)).unsqueeze(0)
    y = torch.tensor(
        [self.cfg.decoder_vocab.to_token_ids(new_example.y)], dtype=torch.long
    )
    return self.encoder_decoder.update_context_state(
        state, src.to(self.device), y.to(self.device)
    ).detached()

  @torch.no_grad()
  @torch.no_grad()
  def decode(
      self,
      query_x: str,
      state: architecture.HybridContextState | None,
      num_samples: int = 1,
      temperature: float = 1.0,
  ) -> torch.Tensor:
    """Batched temperature-sampled decode of query y token ids.

    Args:
      query_x: The query input string.
      state: Optional hybrid state.
      num_samples: Number of sequences to decode in parallel.
      temperature: Sampling temperature.

    Returns:
      (num_samples, L_decode) tensor of token ids (without BOS).
    """
    src = (
        torch.from_numpy(self._encode_text_np(query_x))
        .unsqueeze(0)
        .to(self.device)
    )
    memory, memory_mask = self.encoder_decoder.encode(src)

    # Expand to num_samples (batch_size = 1 -> num_samples)
    expanded_memory = memory.repeat_interleave(num_samples, dim=0)
    expanded_mask = memory_mask.repeat_interleave(num_samples, dim=0)
    if state is not None:
      expanded_state = state.expand_for_queries(num_samples)
    else:
      expanded_state = None

    seq = torch.full(
        (num_samples, 1),
        self.cfg.decoder_vocab.bos_pad_id,
        dtype=torch.long,
        device=self.device,
    )
    vocab_size = len(self.cfg.decoder_vocab.itos)

    for _ in range(self.cfg.decode_len):
      ids_step = seq.cpu()

      curr_mask = torch.zeros(
          (num_samples, vocab_size),
          dtype=torch.float32,
          device="cpu",
      )
      for i in range(num_samples):
        prev_tokens = ids_step[i, 1:].tolist()
        allowed_inds = self.cfg.decoder_vocab.possible_next_token_ids(
            prev_tokens
        )
        if allowed_inds:
          curr_mask[i, allowed_inds] = 1.0

      curr_mask = curr_mask.to(self.device)

      logits = self.encoder_decoder.next_token_logits(
          seq, expanded_memory, expanded_mask, context_state=expanded_state
      )
      # Mask invalid tokens.
      masked_logits = torch.where(
          curr_mask.bool(), logits, torch.tensor(-1e7, device=self.device)
      )

      probs = F.softmax(masked_logits / temperature, dim=-1)
      nxt = torch.multinomial(probs, num_samples=1)
      seq = torch.cat([seq, nxt], dim=1)

    return seq[:, 1:]

  @torch.no_grad()
  def decode_median(
      self,
      query_x: str,
      state: architecture.HybridContextState | None,
      num_samples: int = 128,
      temperature: float = 1.0,
  ) -> float:
    """Decodes num_samples times and returns the median.

    Args:
      query_x: The query input string.
      state: Context state, or None.
      num_samples: Number of independent samples to draw.
      temperature: Sampling temperature.

    Returns:
      Median decoded float value.
    """
    token_ids_batch = self.decode(query_x, state, num_samples, temperature)
    values = np.array([
        self.cfg.decoder_vocab.from_token_ids(token_ids.tolist())[0]
        for token_ids in token_ids_batch
    ])
    return float(np.median(values))


class EpisodicTrainer:
  """Trains a HybridModel with batched episodic (meta-learning) steps."""

  def __init__(
      self,
      model: HybridModel,
      task_sampler: TaskSampler,
      optimizer: optim.Optimizer,
      scheduler: optim.lr_scheduler.LRScheduler | None = None,
      grad_clip_norm: float | None = 1.0,
      grad_accum_steps: int = 1,
  ):
    self.model = model
    self.task_sampler = task_sampler
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.grad_clip_norm = grad_clip_norm
    self.grad_accum_steps = grad_accum_steps
    self.global_step = 0

  def train_step(self) -> dict[str, float]:
    """Runs one batched episodic step with gradient accumulation."""
    self.optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    last_metrics = None
    last_k = 0

    for _ in range(self.grad_accum_steps):
      self.model.train()
      batch = self.task_sampler.sample_batch()
      loss, metrics = self.model.compute_episodic_loss(batch)
      (loss / self.grad_accum_steps).backward()
      accum_loss += metrics["loss_mean"].item()
      last_metrics = metrics
      last_k = batch.k

    if self.grad_clip_norm is not None:
      nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
    self.optimizer.step()
    if self.scheduler is not None:
      self.scheduler.step()

    self.global_step += 1
    assert last_metrics is not None
    avg_loss = accum_loss / self.grad_accum_steps

    return {
        "train_loss": avg_loss,
        "context_size": last_k,
        "num_compressed_pairs": last_metrics["num_compressed_pairs"].item(),
        "learning_rate": self.optimizer.param_groups[0]["lr"],
        "global_step": self.global_step,
    }

  def train(
      self, num_steps: int, log_every: int = 100
  ) -> list[dict[str, float]]:
    """Runs training; returns logged metrics."""
    all_metrics = []
    for step in range(num_steps):
      step_metrics = self.train_step()
      if (step + 1) % log_every == 0:
        logger.info(
            "Step %d/%d: loss=%.4f, k=%d (compressed=%d), lr=%.2e",
            step + 1,
            num_steps,
            step_metrics["train_loss"],
            step_metrics["context_size"],
            step_metrics["num_compressed_pairs"],
            step_metrics["learning_rate"],
        )
        all_metrics.append(step_metrics)
    return all_metrics
