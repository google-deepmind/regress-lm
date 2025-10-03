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

"""Single-process pretraining class for RegressLM."""

import collections
import itertools
import math
import multiprocessing as mp
from regress_lm import core
from regress_lm.pytorch import model as pytorch_model
import torch
from torch import optim
from torch import utils
from torch.optim import lr_scheduler


class Pretrainer:
  """A class to encapsulate the RegressLM pretraining process."""

  def __init__(
      self,
      model: pytorch_model.PyTorchModel,
      learning_rate: float,
      train_ds: utils.data.Dataset[core.Example],
      validation_ds: utils.data.Dataset[core.Example],
      batch_size: int,
      validation_batch_size: int,
      num_epochs: int,
      warmup_steps_fraction: float,
      num_data_workers: int = 0,
      multiprocessing_context: str | mp.context.BaseContext | None = None,
  ):
    """Initialises the Pretrainer with all hyperparameters."""

    self._model = model
    self._optimizer = optim.Adafactor(
        self._model.parameters(), lr=learning_rate
    )

    self._num_epochs = num_epochs
    self._warmup_steps_fraction = warmup_steps_fraction

    # Create dataloaders for training and validation.
    self._train_dl = utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_workers,
        drop_last=False,
        collate_fn=self._model.convert_examples,
        multiprocessing_context=multiprocessing_context,
    )
    self._train_iterator = itertools.cycle(self._train_dl)
    self._val_dl = utils.data.DataLoader(
        dataset=validation_ds,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=num_data_workers,
        collate_fn=self._model.convert_examples,
        multiprocessing_context=multiprocessing_context,
    )

    # Create LR scheduler and global step counter.
    self._scheduler = self._create_lr_scheduler()
    self._global_step = 0

  def _create_lr_scheduler(self) -> lr_scheduler.LambdaLR:
    """Creates a learning rate scheduler."""
    warmup_steps = int(self.total_steps * self._warmup_steps_fraction)

    def lr_lambda(step: int) -> float:
      if step < warmup_steps:
        return step / max(1, warmup_steps)
      progress = (step - warmup_steps) / max(1, self.total_steps - warmup_steps)
      return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_scheduler.LambdaLR(self._optimizer, lr_lambda)

  def run_validation(self) -> dict[str, float]:
    """Runs the validation loop and returns metrics."""
    self._model.eval()
    total_loss, total_items = 0.0, 0
    metric_sums: dict[str, float] = collections.defaultdict(float)

    with torch.no_grad():
      for val_batch in self._val_dl:
        losses_p_ex, metrics = self._model.compute_losses_and_metrics(val_batch)
        bsz = next(iter(val_batch.values())).size(0)

        total_loss += losses_p_ex.sum().item()
        for k, m in metrics.items():
          metric_sums[k] += m.item() * bsz
        total_items += bsz

    val_metrics = {
        f"validation_{k}": v / total_items for k, v in metric_sums.items()
    }
    val_metrics["validation_loss"] = total_loss / total_items
    return val_metrics

  def run_train_step(self) -> dict[str, float]:
    """Runs train step and returns metrics."""
    self._model.train()
    self._optimizer.zero_grad(set_to_none=True)

    batch = next(self._train_iterator)
    losses_per_example, metrics = self._model.compute_losses_and_metrics(batch)
    loss = losses_per_example.mean()
    loss.backward()

    self._optimizer.step()
    self._scheduler.step()
    self._global_step += 1

    train_metrics = {f"train_{k}": v.item() for k, v in metrics.items()}
    train_metrics["train_loss"] = loss.item()
    return train_metrics

  @property
  def global_step(self) -> int:
    return self._global_step

  @property
  def total_steps(self) -> int:
    steps_per_epoch = len(self._train_dl)
    return steps_per_epoch * self._num_epochs
