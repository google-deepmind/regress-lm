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

"""Exponential moving average (EMA) of model parameters."""

from collections.abc import Iterator
import contextlib
from typing import Any
import torch
from torch import nn


def _params(model: nn.Module) -> list[torch.Tensor]:
  return [p.detach() for p in model.parameters() if p.requires_grad]


class ParameterEMA:
  """EMA of a model's trainable parameters, kept in float32.

  As in `tf.train.ExponentialMovingAverage(num_updates=...)`, the decay warms up
  as `min(decay, n / (n + 9))` at the n-th update (from 0): the first update
  copies the parameters. `decay=None` disables it, making all methods no-ops.
  """

  def __init__(self, model: nn.Module, decay: float | None = 0.998):
    if decay is not None and not 0.0 <= decay < 1.0:
      raise ValueError(f"decay ({decay}) must be in [0, 1).")
    self.decay = decay
    self.load(model, {})  # Initializes `shadow` and `num_updates`.

  def update(self, model: nn.Module) -> None:
    """Folds in the current parameters. Call after each optimizer step."""
    if self.decay is None:
      return
    decay = min(self.decay, self.num_updates / (self.num_updates + 9))
    params = [p.float() for p in _params(model)]
    torch._foreach_lerp_(self.shadow, params, 1.0 - decay)  # pylint: disable=protected-access
    self.num_updates += 1

  @contextlib.contextmanager
  def average_parameters(self, model: nn.Module) -> Iterator[None]:
    """Temporarily loads the EMA weights into `model` (no-op if not updated)."""
    params = _params(model) if self.num_updates > 0 else []
    backup = [p.clone() for p in params]
    for p, s in zip(params, self.shadow):
      p.copy_(s)
    try:
      yield
    finally:
      for p, b in zip(params, backup):
        p.copy_(b)

  def save(self, model: nn.Module, checkpoint: dict[str, Any]) -> None:
    """Puts CPU copies of the EMA weights in `checkpoint["model_state"]`."""
    if self.num_updates == 0:  # The EMA weights equal the raw ones.
      return
    checkpoint["train_model_state"] = checkpoint["model_state"]  # To resume.
    checkpoint["ema_num_updates"] = self.num_updates
    with self.average_parameters(model):
      ema = {k: v.to("cpu", copy=True) for k, v in model.state_dict().items()}
    checkpoint["model_state"] = ema  # Downstream loads `model_state`.

  def load(self, model: nn.Module, checkpoint: dict[str, Any]) -> None:
    """Restores the EMA once `model` has loaded `checkpoint["model_state"]`."""
    params = _params(model) if self.decay is not None else []
    self.shadow = [p.float().clone() for p in params]
    self.num_updates = checkpoint.get("ema_num_updates", 0) if params else 0
    if "train_model_state" in checkpoint:  # Raw weights, to resume training.
      model.load_state_dict(checkpoint["train_model_state"])
