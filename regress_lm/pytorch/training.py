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


class Trainer:
  """A class to encapsulate the RegressLM large-scale training process.

  NOTE: The caller is encouraged to still set `train()` or `eval()` explicitly
  on `training_wrapper` to ensure proper synchronization of states.
  """

  def __init__(
      self,
      model: PyTorchModel,
      optimizer_factory: Callable[[Parameters], optim.Optimizer],
      scheduler_factory: Callable[[optim.Optimizer], lr_scheduler.LRScheduler],
      train_ds: utils.data.Dataset[core.Example],
      validation_ds: utils.data.Dataset[core.Example],
      batch_size: int,
      validation_batch_size: int | None = None,
      use_ddp: bool = False,
      num_data_workers: int = 0,
  ):
    self._model = model  # NOTE: Only used as a template if distributed.
    self._training_wrapper = _TrainingModelWrapper(self._model)
    if use_ddp:
      self._training_wrapper = nn.parallel.DistributedDataParallel(
          self._training_wrapper
      )

    self._optimizer = optimizer_factory(self._training_wrapper.parameters())
    self._scheduler = scheduler_factory(self._optimizer)
    self._global_step = 0

    # Create dataloaders for training and validation.
    train_is_iter = isinstance(train_ds, utils.data.IterableDataset)
    if use_ddp and not train_is_iter:
      # No need to split dataset if it's an infinite iterator (e.g. reverb).
      self._train_sampler = DistributedSampler(train_ds)
    else:
      self._train_sampler = None

    self._train_dl = utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=(self._train_sampler is None and not train_is_iter),
        num_workers=num_data_workers,
        drop_last=False,
        sampler=self._train_sampler,
        collate_fn=self._model.converter.convert_examples,
    )
    self._val_dl = utils.data.DataLoader(
        dataset=validation_ds,
        batch_size=validation_batch_size or batch_size,
        num_workers=num_data_workers,
        collate_fn=self._model.converter.convert_examples,
    )

  def run_validation_epoch(self) -> dict[str, float]:
    """Runs the validation loop and returns metrics."""
    self._optimizer.zero_grad(set_to_none=True)  # Free up memory.
    self._training_wrapper.eval()
    total_loss, total_items = 0.0, 0
    metric_sums: dict[str, float] = collections.defaultdict(float)

    with torch.no_grad():
      for val_batch in self._val_dl:
        losses_p_ex, metrics = self.model.compute_losses_and_metrics(val_batch)
        bsz = next(iter(val_batch.values())).size(0)

        total_loss += losses_p_ex.sum().item()
        for k, m in metrics.items():
          metric_sums[k] += m.item() * bsz
        total_items += bsz

    val_metrics = {
        f'validation_{k}': v / total_items for k, v in metric_sums.items()
    }
    val_metrics['validation_loss'] = total_loss / total_items
    return val_metrics

  def run_train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    """Runs train step and returns metrics."""
    self._training_wrapper.train()
    self._optimizer.zero_grad(set_to_none=True)

    losses_per_example, metrics = self._training_wrapper.forward(batch)
    loss = losses_per_example.mean()
    loss.backward()

    self._optimizer.step()
    self._scheduler.step()
    self._global_step += 1

    train_metrics = {f'train_{k}': v.item() for k, v in metrics.items()}
    train_metrics['train_loss'] = loss.item()
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
