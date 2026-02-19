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

"""Custom Pytorch optimizers."""

from torch import optim


class HybridOptimizer(optim.Optimizer):
  """Wraps two optimizers (e.g. Muon + AdamW) behind a single interface."""

  def __init__(
      self, optimizer_1: optim.Optimizer, optimizer_2: optim.Optimizer
  ):
    self.optimizer_1 = optimizer_1
    self.optimizer_2 = optimizer_2
    self.defaults = {}
    self.state = {}  # pytype: disable=annotation-type-mismatch

  @property
  def param_groups(self):
    return self.optimizer_1.param_groups + self.optimizer_2.param_groups

  @param_groups.setter
  def param_groups(self, _):
    pass

  def step(self, closure=None):
    self.optimizer_1.step()
    self.optimizer_2.step()

  def zero_grad(self, set_to_none: bool = True):
    self.optimizer_1.zero_grad(set_to_none=set_to_none)
    self.optimizer_2.zero_grad(set_to_none=set_to_none)

  def state_dict(self):
    return {
        "optimizer_1": self.optimizer_1.state_dict(),
        "optimizer_2": self.optimizer_2.state_dict(),
    }

  def load_state_dict(self, sd):
    self.optimizer_1.load_state_dict(sd["optimizer_1"])
    self.optimizer_2.load_state_dict(sd["optimizer_2"])
