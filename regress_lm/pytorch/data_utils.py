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

"""Torch-specific data utilities for creating efficient, parallelized data pipelines.

This module provides building blocks that follow PyTorch best practices. The
recommended pattern is to use the `ExampleDataset` to wrap raw data and then
pass the model's own `convert_examples` method as the `collate_fn` to the
`DataLoader`.

This separates the responsibility of data fetching (the Dataset) from data
batching and processing (the collate_fn), allowing the DataLoader to
efficiently parallelize the entire pipeline.

Example usage:
```
train_ds = ExampleDataset(examples)
train_dl = utils.data.DataLoader(
    dataset=train_ds,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    collate_fn=self.model.convert_examples,
    pin_memory=True,
)
for batch in train_dl:
  # Do something with the batch
"""

from typing import Sequence

from regress_lm import core
import torch
from torch import utils


class ExampleDataset(utils.data.Dataset):
  """A simple Dataset that holds raw `core.ExampleInput` objects.

  Its responsibility is to simply retrieve an unprocessed example by its index.
  It is intentionally lightweight. The heavy lifting of converting examples
  to tensors is deferred to the DataLoader's `collate_fn`.
  """

  def __init__(self, examples: Sequence[core.ExampleInput]):
    self.examples = examples

  def __len__(self) -> int:
    return len(self.examples)

  def __getitem__(self, idx: int) -> core.ExampleInput:
    return self.examples[idx]


class DictTensorDataset(utils.data.Dataset):
  """Wraps a dictionary of tensors so that each sample is also a dictionary.

  This is useful for small datasets that can be fully pre-processed and held
  in memory. For large datasets, the `ExampleDataset` approach is recommended.
  """

  def __init__(self, tensors: dict[str, torch.Tensor]):
    # Ensure all tensors have the same first dimension (batch size)
    it = iter(tensors.values())
    try:
      first_len = len(next(it))
    except StopIteration as e:
      raise ValueError("tensors dictionary cannot be empty.") from e

    if not all(len(t) == first_len for t in it):
      raise ValueError("All tensors must have the same length.")

    self.tensors = tensors
    self._len = first_len

  def __len__(self) -> int:
    return self._len

  def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    return {k: t[idx] for k, t in self.tensors.items()}
