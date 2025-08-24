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

"""PyTorch implementation of a RegressLM."""

# pytype:disable=attribute-error
from concurrent import futures
import functools
import math
from typing import Sequence
import numpy as np
from regress_lm import core
from regress_lm import vocabs
from regress_lm.models import base as model_base
from regress_lm.models.pytorch import architecture
from regress_lm.models.pytorch import data_utils
import torch
from torch import nn
from torch import optim
from torch import utils
import torch.nn.functional as F

NEG_INF = -1.0e7

# Dict Keys: "encoder_input", "decoder_input", "decoder_target"
Tensor = torch.Tensor


class PyTorchModel(nn.Module, model_base.Model[Tensor]):
  """PyTorch implementation of a RegressLM."""

  def __init__(
      self,
      encoder_vocab: vocabs.EncoderVocab[str],
      decoder_vocab: vocabs.DecoderVocab[float],
      max_input_len: int = 2048,
      max_num_objs: int = 1,
      z_loss_coef: float | None = None,
      compile_model: bool = True,  # Turn off for tests.
      **architecture_kwargs,
  ):
    super().__init__()
    self.max_input_len = max_input_len
    self.max_num_objs = max_num_objs
    self.z_loss_coef = z_loss_coef

    self.encoder_vocab = encoder_vocab
    self.decoder_vocab = decoder_vocab

    self.encoder_decoder = architecture.EncoderDecoder(
        encoder_vocab_size=len(self.encoder_vocab),
        decoder_vocab_size=len(self.decoder_vocab),
        encoder_pad_idx=self.encoder_vocab.pad_id,
        max_encoder_len=self.max_input_len,
        max_decoder_len=self.decode_len + 1,
        **architecture_kwargs,
    )

    if compile_model:
      self.encoder_decoder = torch.compile(self.encoder_decoder)

  @property
  def device(self) -> torch.device:
    return next(self.parameters()).device

  def to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v.to(self.device) for k, v in batch.items()}

  @property
  def decode_len(self) -> int:
    return self.max_num_objs * self.decoder_vocab.num_tokens_per_obj

  def compute_losses_and_metrics(
      self, examples: dict[str, Tensor]
  ) -> tuple[Tensor, dict[str, Tensor]]:
    examples = self.to_device(examples)
    metrics = {}
    logits = self.encoder_decoder.forward(
        examples['encoder_input'], examples['decoder_input']
    )
    targets = examples['decoder_target']

    ce_loss_per_tokens = F.cross_entropy(
        logits.permute(0, 2, 1),
        targets,
        ignore_index=self.decoder_vocab.bos_pad_id,
        reduction='none',
    )
    loss_per_example = ce_loss_per_tokens.sum(dim=1)

    if self.z_loss_coef is not None:
      log_z = torch.logsumexp(logits, dim=-1)
      z_loss_per_token = self.z_loss_coef * (log_z**2)
      loss_mask = (targets != self.decoder_vocab.bos_pad_id).float()
      z_loss_per_example = (z_loss_per_token * loss_mask).sum(dim=1)
      loss_per_example = loss_per_example + z_loss_per_example

    metrics['loss_mean'] = loss_per_example.mean().detach()
    return loss_per_example, metrics

  @torch.no_grad()
  def decode(
      self,
      inputs: dict[str, Tensor],
      num_samples: int,
      temperature: float = 1.0,
  ) -> tuple[Tensor, np.ndarray]:
    self.encoder_decoder.eval()
    inputs = self.to_device(inputs)

    encoder_input = inputs['encoder_input']  # (B, L_src)
    batch_size = encoder_input.shape[0]
    # memory: (B, L_src, D_model), memory_key_padding_mask: (B, L_src)
    memory, memory_key_padding_mask = self.encoder_decoder.encode(encoder_input)

    # Expand encoder outputs and masks for num_samples
    # Effectively, new batch_size = B * num_samples
    # memory: (B, L_src, D) -> (B, 1, L_src, D) -> (B, S, L_src, D)
    # -> (B*S, L_src, D)
    expanded_memory = (
        memory.unsqueeze(1)
        .expand(-1, num_samples, -1, -1)
        .reshape(batch_size * num_samples, memory.size(1), memory.size(2))
    )
    expanded_memory_key_padding_mask = (
        memory_key_padding_mask.unsqueeze(1)
        .expand(-1, num_samples, -1)
        .reshape(batch_size * num_samples, memory_key_padding_mask.size(1))
    )

    # Initialize decoder input for the expanded batch, start with <pad>.
    current_tgt_ids = torch.full(
        (batch_size * num_samples, 1),
        self.decoder_vocab.bos_pad_id,
        dtype=torch.long,
        device=self.device,
    )

    # Store all generated token IDs for all sequences in the expanded batch
    generated_sequences_ids = torch.zeros(
        (batch_size * num_samples, self.decode_len),
        dtype=torch.long,
        device=self.device,
    )

    # Batched autoregressive decoding loop
    expanded_batch_size = batch_size * num_samples

    def get_allowed_tokens(i: int, step_idx: int) -> list[int]:
      prev_tokens = generated_sequences_ids[i, :step_idx].tolist()
      return self.decoder_vocab.possible_next_token_ids(prev_tokens)

    for step_idx in range(self.decode_len):
      # Get logits for the next token for all (B * num_samples) sequences
      # Shape: (B*S, V)
      logits = self.encoder_decoder.next_token_logits(
          current_tgt_ids, expanded_memory, expanded_memory_key_padding_mask
      )

      # Apply constraints using online computed mask
      curr_mask = torch.zeros(
          (expanded_batch_size, len(self.decoder_vocab)), device=self.device
      )
      parallel_fn = functools.partial(get_allowed_tokens, step_idx=step_idx)
      with futures.ThreadPoolExecutor() as executor:
        results = executor.map(parallel_fn, range(expanded_batch_size))

      for i, allowed_inds in enumerate(results):
        curr_mask[i, allowed_inds] = 1.0

      masked_logits = (1.0 - curr_mask) * NEG_INF + curr_mask * logits

      # Apply temperature sampling, 1 token for each of the B*S sequences
      probs = F.softmax(masked_logits / temperature, dim=-1)
      token_ids = torch.multinomial(probs, num_samples=1)  # (B*S, 1)
      # Store the predicted token IDs
      generated_sequences_ids[:, step_idx] = token_ids.squeeze(-1)

      # Prepare input for the next step, but only if not the last float token
      if step_idx < self.decode_len - 1:
        current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

    # Reshape outputs back to (B, num_samples, L_decode)
    final_decoded_ids = generated_sequences_ids.view(
        batch_size, num_samples, self.decode_len
    )

    # Compute equivalent floats.
    output_floats = np.empty(
        (batch_size, num_samples, self.max_num_objs), dtype=object
    )
    for b in range(batch_size):
      for s_idx in range(num_samples):
        output_floats[b, s_idx, :] = self.decoder_vocab.from_token_ids(
            final_decoded_ids[b, s_idx, :].tolist()
        )

    return final_decoded_ids, output_floats

  def log_prob(self, examples: dict[str, Tensor]) -> Tensor:
    self.encoder_decoder.eval()
    examples = self.to_device(examples)
    enc_input = examples['encoder_input']
    dec_input = examples['decoder_input']
    dec_target = examples['decoder_target']

    logits = self.encoder_decoder.forward(enc_input, dec_input)
    log_probs = F.log_softmax(logits, dim=-1)

    true_log_probs = torch.gather(
        log_probs, dim=2, index=dec_target.unsqueeze(-1)
    )

    pad_mask = dec_target != self.decoder_vocab.bos_pad_id
    true_log_probs_masked = true_log_probs.squeeze(-1) * pad_mask
    sequence_sum_log_probs = true_log_probs_masked.sum(dim=1)
    return sequence_sum_log_probs

  def convert_inputs(
      self, inputs: Sequence[core.ExampleInput]
  ) -> dict[str, Tensor]:
    strings = [example.x for example in inputs]
    encoder_token_ids = [self.encoder_vocab.to_token_ids(s) for s in strings]
    encoder_input = [self._pad_or_truncate(t) for t in encoder_token_ids]
    return {'encoder_input': torch.tensor(encoder_input)}

  def convert_examples(
      self, examples: Sequence[core.Example]
  ) -> dict[str, Tensor]:
    y_values = [example.y for example in examples]
    y_tokens_list = [self.decoder_vocab.to_token_ids(y) for y in y_values]

    decoder_inputs = []
    decoder_targets = []
    pad_id = self.decoder_vocab.bos_pad_id

    for t in y_tokens_list:
      padding_needed = self.decode_len - len(t)
      # Input: [pad, t_1, ..., t_n, pad, ..., pad]
      decoder_inputs.append([pad_id] + t + [pad_id] * padding_needed)
      # Target: [t_1, ..., t_n, pad, ..., pad]
      decoder_targets.append(t + [pad_id] * (padding_needed + 1))

    decoder_out = {
        'decoder_input': torch.tensor(decoder_inputs),
        'decoder_target': torch.tensor(decoder_targets),
    }
    return self.convert_inputs(examples) | decoder_out

  def _pad_or_truncate(self, token_ids: list[int]) -> list[int]:
    encoder_pad_idx = self.encoder_vocab.pad_id
    if len(token_ids) > self.max_input_len:
      return token_ids[: self.max_input_len]
    return token_ids + [encoder_pad_idx] * (self.max_input_len - len(token_ids))


class _EarlyStoppingTracker:
  """Tracks valid loss, saves the best model state, and detects overfitting."""

  def __init__(self, patience: int | None = None):
    """Initializes the tracker.

    Args:
      patience: The number of epochs to wait for improvement before stopping. If
        None, early stopping is disabled.
    """
    self.patience = patience
    self.best_loss = float('inf')
    self.best_state = None
    # Counter for epochs without improvement.
    self.epochs_without_improvement = 0

  def update(self, current_loss: float, model: nn.Module):
    """Updates the tracker with the latest validation loss."""
    if current_loss < self.best_loss:
      # We have a new best loss, so we save the model and reset the counter.
      self.best_loss = current_loss
      self.best_state = model.state_dict()
      self.epochs_without_improvement = 0
    else:
      # The loss did not improve, so we increment the counter.
      self.epochs_without_improvement += 1

  def should_stop(self) -> bool:
    """Returns True if training should stop, based on the patience counter."""
    if self.patience is None:  # Early stopping is disabled.
      return False
    return self.epochs_without_improvement >= self.patience


class PyTorchFineTuner(model_base.FineTuner):
  """PyTorch implementation of a local finetuner."""

  def __init__(
      self,
      model: PyTorchModel,
      optimizer: optim.Optimizer,
      *,
      max_epochs: int = 100,
      batch_size: int | None = None,
      batch_size_per_device: int | None = None,
      patience: int | None = 1,
  ):
    """Initializes the fine-tuner.

    Args:
      model: The PyTorch model to be fine-tuned.
      optimizer: An optional PyTorch optimizer.
      max_epochs: The maximum number of epochs to train for.
      batch_size: The desired *effective* batch size. If None, performs
        full-batch gradient descent (uses the entire dataset as one batch).
      batch_size_per_device: The maximum batch size that can fit on the GPU. If
        the effective `batch_size` is larger than this, gradient accumulation
        will be used automatically.
      patience: The number of epochs to wait for improvement before early
        stopping. If None, early stopping is disabled.
    """
    self.model = model
    self.optimizer = optimizer
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.batch_size_per_device = batch_size_per_device
    self.patience = patience

  def fine_tune(
      self,
      examples: Sequence[core.Example],
      validation_examples: Sequence[core.Example] | None = None,
      seed: int | None = None,
  ) -> None:
    """Fine-tunes the model using the provided examples."""
    validation_examples = validation_examples or examples

    g = torch.Generator()
    if seed is not None:
      g.manual_seed(seed)

    effective_batch_size = self.batch_size or len(examples)
    micro_batch_size = effective_batch_size
    if self.batch_size_per_device is not None:
      micro_batch_size = min(effective_batch_size, self.batch_size_per_device)
    grad_acc_steps = math.ceil(effective_batch_size / micro_batch_size)

    train_tensors = self.model.convert_examples(examples)
    valid_tensors = self.model.convert_examples(validation_examples)

    train_dl = utils.data.DataLoader(
        data_utils.DictTensorDataset(train_tensors),
        batch_size=micro_batch_size,
        shuffle=True,
        generator=g,
    )
    valid_dl = utils.data.DataLoader(
        data_utils.DictTensorDataset(valid_tensors),
        batch_size=micro_batch_size,
        shuffle=False,
    )

    # Perform an initial validation run before training
    tracker = _EarlyStoppingTracker(patience=self.patience)
    initial_val_loss = self._run_validation_epoch(valid_dl)
    tracker.update(initial_val_loss, self.model)

    for _ in range(self.max_epochs):
      self._run_training_epoch(train_dl, grad_acc_steps)
      val_loss = self._run_validation_epoch(valid_dl)
      tracker.update(val_loss, self.model)
      if tracker.should_stop():
        break

    if tracker.best_state:
      self.model.load_state_dict(tracker.best_state)

  def _run_training_epoch(
      self, train_dl: utils.data.DataLoader, grad_acc_steps: int
  ):
    """Runs a single training epoch with gradient accumulation."""
    self.model.train()
    self.optimizer.zero_grad()

    for i, batch in enumerate(train_dl):
      self._forward_backward_pass(batch, grad_acc_steps)

      if (i + 1) % grad_acc_steps == 0:
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Final step for any remaining gradients
    if len(train_dl) % grad_acc_steps != 0:
      self.optimizer.step()
      self.optimizer.zero_grad()

  def _forward_backward_pass(
      self, batch: dict[str, torch.Tensor], grad_acc_steps: int
  ) -> None:
    losses_per_example, _ = self.model.compute_losses_and_metrics(batch)
    loss = losses_per_example.mean()
    scaled_loss = loss / grad_acc_steps
    scaled_loss.backward()

  def _run_validation_epoch(self, valid_dl: utils.data.DataLoader) -> float:
    self.model.eval()
    all_val_losses = []

    with torch.no_grad():
      for valid_batch in valid_dl:
        batch_losses, _ = self.model.compute_losses_and_metrics(valid_batch)
        all_val_losses.extend(batch_losses.cpu().numpy())

    return float(np.mean(all_val_losses))
