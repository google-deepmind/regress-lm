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

"""Pretraining class for RegressLM."""

import collections
from typing import Callable, Iterator
from regress_lm import core
from regress_lm.pytorch import model as pytorch_model
import torch
from torch import nn
from torch import optim
from torch import utils
from torch.optim import lr_scheduler

DistributedSampler = torch.utils.data.distributed.DistributedSampler
PyTorchModel = pytorch_model.PyTorchModel


class DistributedTrainingWrapper(nn.Module):
  """A wrapper for the PyTorchModel that enables distributed training.

  nn.parallel.DistributedDataParallel only activates when calling forward().
  """

  def __init__(self, model: PyTorchModel):
    super().__init__()
    self.model = model

  def forward(
      self, batch: dict[str, torch.Tensor]
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return self.model.compute_losses_and_metrics(batch)


Parameters = Iterator[nn.Parameter]


class Pretrainer:
  """A class to encapsulate the RegressLM pretraining process."""

  def __init__(
      self,
      model: PyTorchModel,
      optimizer_factory: Callable[[Parameters], optim.Optimizer],
      scheduler_factory: Callable[[optim.Optimizer], lr_scheduler.LRScheduler],
      train_ds: utils.data.Dataset[core.Example],
      validation_ds: utils.data.Dataset[core.Example],
      batch_size: int,
      validation_batch_size: int | None = None,
      distributed_rank: int | None = None,
      num_data_workers: int = 0,
  ):
    self._model = model

    self._ddp_model = DistributedTrainingWrapper(model)
    if distributed_rank is not None:
      self._ddp_model = nn.parallel.DistributedDataParallel(
          self._ddp_model, device_ids=[distributed_rank]
      )

    self._optimizer = optimizer_factory(self._ddp_model.parameters())
    self._scheduler = scheduler_factory(self._optimizer)
    self._global_step = 0

    # Create dataloaders for training and validation.
    self._train_sampler = (
        DistributedSampler(train_ds) if distributed_rank is not None else None
    )
    self._train_dl = utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=(self._train_sampler is None),
        num_workers=num_data_workers,
        drop_last=False,
        sampler=self._train_sampler,
        collate_fn=self._model.converter.convert_examples,
    )
    self._val_dl = utils.data.DataLoader(
        dataset=validation_ds,
        batch_size=validation_batch_size or batch_size,
        shuffle=False,
        num_workers=num_data_workers,
        collate_fn=self._model.converter.convert_examples,
    )

  def run_validation_epoch(self) -> dict[str, float]:
    """Runs the validation loop and returns metrics."""
    self._ddp_model.eval()
    total_loss, total_items = 0.0, 0
    metric_sums: dict[str, float] = collections.defaultdict(float)

    with torch.no_grad():
      for val_batch in self._val_dl:
        losses_p_ex, metrics = self._ddp_model.forward(val_batch)
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

  def run_train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    """Runs train step and returns metrics."""
    self._ddp_model.train()
    self._optimizer.zero_grad(set_to_none=True)

    losses_per_example, metrics = self._ddp_model.forward(batch)
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
  def train_sampler(self) -> DistributedSampler | None:
    return self._train_sampler

  @property
  def train_dl(self) -> utils.data.DataLoader:
    return self._train_dl

  @property
  def optimizer(self) -> optim.Optimizer:
    return self._optimizer

  @property
  def scheduler(self) -> lr_scheduler.LRScheduler:
    return self._scheduler
