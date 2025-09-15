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

"""PyTorch implementation of a local finetuner."""

import math
from typing import Sequence, cast
import numpy as np
from regress_lm import core
from regress_lm.models import base as model_base
from regress_lm.models.pytorch import data_utils
import torch
from torch import nn
from torch import optim
from torch import utils


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
      model: model_base.Model[torch.Tensor],
      optimizer: optim.Optimizer,
      *,
      max_epochs: int = 100,
      batch_size: int | None = None,
      batch_size_per_device: int | None = None,
      patience: int | None = 1,
  ):
    """Initializes the fine-tuner.

    Args:
      model: The PyTorch model / nn.Module to be fine-tuned.
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
    self.model = cast(nn.Module, model)
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
