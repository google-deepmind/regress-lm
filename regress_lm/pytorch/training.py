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
import numpy as np
from regress_lm import core
from regress_lm.pytorch import model as pytorch_model
import torch
from torch import nn
from torch import optim
from torch import utils
import torch.distributed as dist
from torch.optim import lr_scheduler

DistributedSampler = torch.utils.data.distributed.DistributedSampler
PyTorchModel = pytorch_model.PyTorchModel


class _TrainingModelWrapper(nn.Module):
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


def get_sampler(
    ds: utils.data.Dataset[core.Example], use_ddp: bool
) -> DistributedSampler[core.Example] | None:
  """Returns a sampler for the dataset, if applicable."""
  # TODO: Check should be specifically for reverb or a random iterator.
  # Doesn't apply to static iterables, which would lead to duplicate rank work.
  if use_ddp and not isinstance(ds, utils.data.IterableDataset):
    # No need to split dataset if it's an infinite iterator (e.g. reverb).
    return DistributedSampler(ds)
  else:
    return None


class Trainer:
  """RegressLM large-scale training process, including distributed cases."""

  def __init__(
      self,
      model: PyTorchModel,
      optimizer_factory: Callable[[Parameters], optim.Optimizer],
      scheduler_factory: Callable[[optim.Optimizer], lr_scheduler.LRScheduler],
      train_ds: utils.data.Dataset[core.Example],
      validation_ds: utils.data.Dataset[core.Example],
      batch_size: int,
      validation_batch_size: int | None = None,
      grad_acc_steps: int = 1,
      use_ddp: bool = False,
      num_data_workers: int = 0,
  ):
    self._model = model  # NOTE: Only used as a template if distributed.
    self._training_wrapper = _TrainingModelWrapper(self._model)
    self._grad_acc_steps = grad_acc_steps
    self._use_ddp = use_ddp
    if self._use_ddp:
      self._training_wrapper = nn.parallel.DistributedDataParallel(
          self._training_wrapper
      )

    self._optimizer = optimizer_factory(self._training_wrapper.parameters())
    self._scheduler = scheduler_factory(self._optimizer)
    self._global_step = 0

    # Create dataloaders for training and validation.
    train_is_iter = isinstance(train_ds, utils.data.IterableDataset)
    self._train_sampler = get_sampler(train_ds, self._use_ddp)

    self._train_dl = utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=(self._train_sampler is None and not train_is_iter),
        sampler=self._train_sampler,
        num_workers=num_data_workers,
        collate_fn=self._model.converter.convert_examples,
        drop_last=False,
    )

    self._val_dl = utils.data.DataLoader(
        dataset=validation_ds,
        batch_size=validation_batch_size or batch_size,
        shuffle=False,
        sampler=get_sampler(validation_ds, self._use_ddp),
        num_workers=num_data_workers,
        collate_fn=self._model.converter.convert_examples,
    )

  def run_validation_epoch(self) -> dict[str, float]:
    """Runs the validation loop and returns metrics."""
    self._optimizer.zero_grad(set_to_none=True)  # Free up memory.
    self._training_wrapper.eval()

    metric_sums: dict[str, torch.Tensor] = collections.defaultdict(
        lambda: torch.tensor(0.0)
    )
    total_items = torch.tensor(0)

    with torch.no_grad():
      for val_batch in self._val_dl:
        _, metrics = self._training_wrapper.forward(val_batch)
        bsz = next(iter(val_batch.values())).size(0)

        for key, value_tensor in metrics.items():
          metric_sums[key] += value_tensor.to('cpu') * bsz
        total_items += bsz

    if self._use_ddp:  # Move final sums to GPU for communication.
      total_items = total_items.to(self._model.device)
      dist.all_reduce(total_items, op=dist.ReduceOp.SUM)
      for key in metric_sums:
        metric_sums[key] = metric_sums[key].to(self._model.device)
        dist.all_reduce(metric_sums[key], op=dist.ReduceOp.SUM)

    val_metrics = {}
    mean_val_loss = metric_sums['loss_mean'] / total_items
    val_metrics['validation_loss'] = mean_val_loss.item()
    val_metrics['validation_perplexity'] = torch.exp(mean_val_loss).item()
    for key, value in metric_sums.items():
      if key != 'loss_mean':
        val_metrics[f'validation_{key}'] = (value / total_items).item()
    return val_metrics

  def run_train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    """Runs train step and returns metrics."""
    self._training_wrapper.train()

    losses_per_example, metrics = self._training_wrapper.forward(batch)
    loss = losses_per_example.mean()
    loss /= self._grad_acc_steps
    loss.backward()

    if (self._global_step + 1) % self._grad_acc_steps == 0:
      self._optimizer.step()
      self._scheduler.step()
      self._optimizer.zero_grad(set_to_none=True)

    self._global_step += 1

    if self._use_ddp:
      for k in metrics:
        dist.all_reduce(metrics[k], op=dist.ReduceOp.SUM)
        metrics[k] /= dist.get_world_size()

    train_metrics = {f'train_{k}': v.item() for k, v in metrics.items()}
    train_metrics['train_perplexity'] = np.exp(train_metrics['train_loss_mean'])
    train_metrics['train_loss'] = loss.item() * self._grad_acc_steps
    return train_metrics

  def save_checkpoint(self, checkpoint_path: str) -> None:
    """Saves the current training state to a checkpoint file."""
    state = {
        'model_state': self.model.state_dict(),
        'optimizer_state': self._optimizer.state_dict(),
        'scheduler_state': self._scheduler.state_dict(),
        'global_step': self._global_step,
    }
    torch.save(state, checkpoint_path)

  def load_checkpoint(self, checkpoint_path: str) -> None:
    """Loads the training state from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    self.model.load_state_dict(checkpoint['model_state'])
    self._optimizer.load_state_dict(checkpoint['optimizer_state'])
    self._scheduler.load_state_dict(checkpoint['scheduler_state'])
    self._global_step = checkpoint['global_step']

  @property
  def global_step(self) -> int:
    return self._global_step

  @property
  def train_sampler(self) -> DistributedSampler[core.Example] | None:
    return self._train_sampler

  @property
  def train_dl(self) -> utils.data.DataLoader[core.Example]:
    return self._train_dl

  @property
  def optimizer(self) -> optim.Optimizer:
    return self._optimizer

  @property
  def scheduler(self) -> lr_scheduler.LRScheduler:
    return self._scheduler

  @property
  def model(self) -> PyTorchModel:
    """Returns the original model, accounting for distributed wrapping."""
    if isinstance(self._training_wrapper, nn.parallel.DistributedDataParallel):
      # We must return it from wrapper's module, which contains true weights.
      return self._training_wrapper.module.model  # pytype:disable=attribute-error
    else:
      return self._model

  @property
  def training_wrapper(self) -> nn.Module:
    return self._training_wrapper

  @property
  def current_epoch(self) -> int | None:
    if isinstance(self._train_dl.dataset, utils.data.IterableDataset):
      return None
    return self._global_step // len(self.train_dl)
