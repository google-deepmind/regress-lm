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

"""Episodic pretraining for IncrementalEncoderDecoder.

Each training step samples an episode:
  - k context (x,y) pairs from the same task
  - 1+ query (x,y) pairs from the same task

The context pairs are encoded into a linear attention state S.
The decoder predicts query y tokens using both the query encoder memory
(full softmax cross-attention) and the context state S (linear cross-attention).
"""

import dataclasses
import logging
import random
from typing import Any, Callable, Sequence

import numpy as np
from regress_lm import core
from regress_lm import vocabs
from regress_lm.pytorch.online import architecture as incremental_architecture
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

Tensor = torch.Tensor
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Episode:
  """A single training episode with context and query from the same task.

  Attributes:
    context_examples: The (x,y) pairs used to build the context state S.
    query_examples: The (x,y) pairs the model must predict given the context.
  """

  context_examples: list[core.Example]
  query_examples: list[core.Example]


class TaskSampler:
  """Samples episodes from a collection of tasks.

  Each task is a callable that returns (x, y) pairs. During training,
  the sampler picks a task, draws k context pairs and n_query query pairs
  from it, forming an Episode.
  """

  def __init__(
      self,
      tasks: Sequence[Callable[[int], list[core.Example]]],
      max_context_size: int = 5,
      num_queries_per_episode: int = 1,
  ):
    """Initializes the task sampler.

    Args:
      tasks: A sequence of callables. Each task(n) returns n Example instances
        sampled from that task's (x,y) distribution.
      max_context_size: Maximum number of context pairs per episode. The actual
        k is sampled uniformly from [0, max_context_size] each episode.
      num_queries_per_episode: Number of query pairs per episode.
    """
    self.tasks = tasks
    self.max_context_size = max_context_size
    self.num_queries_per_episode = num_queries_per_episode

  def sample_episode(self) -> Episode:
    """Samples a single episode."""
    task = random.choice(self.tasks)
    k = random.randint(0, self.max_context_size)
    total_needed = k + self.num_queries_per_episode
    examples = task(total_needed)
    return Episode(
        context_examples=examples[:k],
        query_examples=examples[k:],
    )


@dataclasses.dataclass(frozen=True)
class IncrementalModelConfig:
  """Configuration for an IncrementalModel.

  Attributes:
    encoder_vocab: Vocabulary for encoding text inputs x.
    decoder_vocab: Vocabulary for decoding float outputs y.
    max_input_len: Maximum encoder input length.
    architecture_kwargs: Kwargs passed to IncrementalEncoderDecoder.
    max_num_objs: Number of float objects per example.
    z_loss_coef: Optional z-loss coefficient for logit regularization.
  """

  encoder_vocab: vocabs.EncoderVocab[str]
  decoder_vocab: vocabs.DecoderVocab[float]
  max_input_len: int
  architecture_kwargs: dict[str, Any]
  max_num_objs: int = 1
  z_loss_coef: float | None = None

  @property
  def decode_len(self) -> int:
    """Total number of decoder tokens per example."""
    return self.max_num_objs * self.decoder_vocab.num_tokens_per_obj


class IncrementalModel(nn.Module):
  """Wraps IncrementalEncoderDecoder with episodic training and state APIs.

  This model supports:
    1. Episodic training: build context state from (x,y) pairs, predict query y.
    2. Incremental inference: build_state / update_state / decode_from_state.
  """

  def __init__(self, config: IncrementalModelConfig):
    super().__init__()
    self.cfg = config
    self.encoder_decoder = incremental_architecture.IncrementalEncoderDecoder(
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

  def _to_device(self, t: Tensor) -> Tensor:
    return t.to(self.device, non_blocking=True)

  def _encode_text(self, text: str) -> Tensor:
    """Tokenizes and pads a single text string to encoder input tensor."""
    token_ids = self.cfg.encoder_vocab.to_token_ids(text)
    padded = np.full(
        self.cfg.max_input_len, self.cfg.encoder_vocab.pad_id, dtype=np.int64
    )
    insert_len = min(len(token_ids), self.cfg.max_input_len)
    padded[:insert_len] = token_ids[:insert_len]
    return torch.from_numpy(padded)

  def _encode_y(self, y: float | Sequence[float]) -> Tensor:
    """Tokenizes a y value to decoder token ids."""
    return torch.tensor(
        self.cfg.decoder_vocab.to_token_ids(y), dtype=torch.long
    )

  def _prepare_context_tensors(
      self,
      context_examples: Sequence[core.Example],
  ) -> tuple[list[Tensor], list[Tensor]]:
    """Prepares batched context tensors from examples.

    Args:
      context_examples: Context (x,y) pairs.

    Returns:
      Tuple of (context_src_list, context_y_tokens_list).
    """
    src_list = []
    y_tokens_list = []
    for ex in context_examples:
      src = self._encode_text(ex.x).unsqueeze(0)  # (1, L)
      y_tokens = self._encode_y(ex.y).unsqueeze(0)  # (1, T_y)
      src_list.append(self._to_device(src))
      y_tokens_list.append(self._to_device(y_tokens))
    return src_list, y_tokens_list

  def compute_episodic_loss(
      self, episode: Episode
  ) -> tuple[Tensor, dict[str, Tensor]]:
    """Computes loss for a single episode.

    Args:
      episode: An Episode with context and query examples.

    Returns:
      Tuple of (loss, metrics_dict).
    """
    # Build context state from context pairs.
    context_state = None
    if episode.context_examples:
      src_list, y_tokens_list = self._prepare_context_tensors(
          episode.context_examples
      )
      context_state = self.encoder_decoder.build_context_state(
          src_list, y_tokens_list
      )

    # Compute loss over query examples.
    total_loss = torch.tensor(0.0, device=self.device)
    for query_ex in episode.query_examples:
      # Prepare query encoder input.
      query_src = self._to_device(
          self._encode_text(query_ex.x).unsqueeze(0)
      )  # (1, L)

      # Prepare decoder input/target.
      y_tokens = self.cfg.decoder_vocab.to_token_ids(query_ex.y)
      dec_len = self.cfg.decode_len
      pad_id = self.cfg.decoder_vocab.bos_pad_id

      dec_input = np.full(dec_len + 1, pad_id, dtype=np.int64)
      dec_target = np.full(dec_len + 1, pad_id, dtype=np.int64)
      insert_len = min(len(y_tokens), dec_len)
      dec_input[1 : insert_len + 1] = y_tokens[:insert_len]
      dec_target[:insert_len] = y_tokens[:insert_len]

      dec_input_t = self._to_device(
          torch.from_numpy(dec_input).unsqueeze(0)
      )  # (1, dec_len+1)
      dec_target_t = self._to_device(
          torch.from_numpy(dec_target).unsqueeze(0)
      )  # (1, dec_len+1)

      # Forward pass.
      logits = self.encoder_decoder.forward(
          query_src, dec_input_t, context_state=context_state
      )  # (1, dec_len+1, vocab_size)

      # Cross-entropy loss over non-padded tokens.
      ce_loss = F.cross_entropy(
          logits.squeeze(0),  # (dec_len+1, vocab_size)
          dec_target_t.squeeze(0),  # (dec_len+1,)
          ignore_index=pad_id,
      )
      total_loss = total_loss + ce_loss

    avg_loss = total_loss / len(episode.query_examples)
    metrics = {'loss_mean': avg_loss.detach()}
    return avg_loss, metrics

  # ── Inference APIs ──────────────────────────────────────────────────

  @torch.no_grad()
  def build_state(
      self, examples: Sequence[core.Example]
  ) -> incremental_architecture.ContextState:
    """Builds context state from examples. No gradient, for inference."""
    src_list, y_tokens_list = self._prepare_context_tensors(examples)
    return self.encoder_decoder.build_context_state(src_list, y_tokens_list)

  @torch.no_grad()
  def update_state(
      self,
      state: incremental_architecture.ContextState,
      new_example: core.Example,
  ) -> incremental_architecture.ContextState:
    """Incrementally adds one (x,y) pair to the state. O(L * d^2)."""
    src = self._to_device(self._encode_text(new_example.x).unsqueeze(0))
    y_tokens = self._to_device(self._encode_y(new_example.y).unsqueeze(0))
    return self.encoder_decoder.update_context_state(state, src, y_tokens)


class EpisodicTrainer:
  """Trains an IncrementalModel using episodic (meta-learning) training.

  Each training step:
    1. Sample an episode (k context pairs + query pairs from same task).
    2. Build context state S from context pairs.
    3. Predict query y tokens using both query memory and S.
    4. Backprop through everything, including state construction.
  """

  def __init__(
      self,
      model: IncrementalModel,
      task_sampler: TaskSampler,
      optimizer: optim.Optimizer,
      scheduler: optim.lr_scheduler.LRScheduler | None = None,
  ):
    self.model = model
    self.task_sampler = task_sampler
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.global_step = 0

  def train_step(self) -> dict[str, float]:
    """Runs one episodic training step.

    Returns:
      Dictionary of training metrics.
    """
    self.model.train()
    self.global_step += 1

    episode = self.task_sampler.sample_episode()
    loss, metrics = self.model.compute_episodic_loss(episode)

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    self.optimizer.step()
    if self.scheduler is not None:
      self.scheduler.step()

    return {
        'train_loss': metrics['loss_mean'].item(),
        'context_size': len(episode.context_examples),
        'learning_rate': self.optimizer.param_groups[0]['lr'],
        'global_step': self.global_step,
    }

  def train(
      self,
      num_steps: int,
      log_every: int = 100,
  ) -> list[dict[str, float]]:
    """Runs episodic training for num_steps steps.

    Args:
      num_steps: Total number of training steps.
      log_every: Log metrics every this many steps.

    Returns:
      List of metric dictionaries, one per logged step.
    """
    all_metrics = []
    for step in range(num_steps):
      step_metrics = self.train_step()

      if (step + 1) % log_every == 0:
        logger.info(
            'Step %d/%d: loss=%.4f, k=%d, lr=%.2e',
            step + 1,
            num_steps,
            step_metrics['train_loss'],
            step_metrics['context_size'],
            step_metrics['learning_rate'],
        )
        all_metrics.append(step_metrics)

    return all_metrics
