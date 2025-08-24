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

"""Torch-specific data utils."""

import torch
from torch import utils


class DictTensorDataset(utils.data.Dataset):
  """Wraps a dict-of-Tensors so that each sample is still a dict."""

  def __init__(self, tensors: dict[str, torch.Tensor]):
    self._keys = list(tensors.keys())
    self._tensors = list(tensors.values())

  def __len__(self) -> int:
    return self._tensors[0].shape[0]

  def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    return {k: t[idx] for k, t in zip(self._keys, self._tensors)}
