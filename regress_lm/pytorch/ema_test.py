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

"""Tests for the parameter EMA."""

from regress_lm.pytorch import ema as ema_lib
import torch
from torch import nn

from absl.testing import absltest


class ParameterEMATest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = nn.Linear(1, 1)

  def _fill(self, value: float) -> None:
    for p in self.model.parameters():
      nn.init.constant_(p, value)

  def _assert_params(self, *expected: float) -> None:
    params = nn.utils.parameters_to_vector(self.model.parameters())
    torch.testing.assert_close(params, torch.tensor(expected))

  def test_update_and_average(self):
    self.model.bias.requires_grad_(False)  # Frozen, so not averaged.
    ema = ema_lib.ParameterEMA(self.model, decay=0.15)
    for value in (1.0, 3.0, 5.0):
      self._fill(value)
      ema.update(self.model)
    # Decays 0 (copy), 1/10 (warmup), then 0.15 (capped, instead of 2/11).
    with ema.average_parameters(self.model):
      self._assert_params(0.15 * (0.1 * 1.0 + 0.9 * 3.0) + 0.85 * 5.0, 5.0)
    self._assert_params(5.0, 5.0)  # Restored.

  def test_float32_average(self):
    self.model.to(torch.bfloat16)
    ema = ema_lib.ParameterEMA(self.model, decay=0.9)
    for value in (1.0, 2.0):
      self._fill(value)
      ema.update(self.model)
    # 1.9 (= 0.1 * 1.0 + 0.9 * 2.0) would round to 1.8984375 in bfloat16.
    torch.testing.assert_close(ema.shadow[0], torch.tensor([[1.9]]))

  def test_noop_if_disabled_or_not_updated(self):
    disabled = ema_lib.ParameterEMA(self.model, decay=None)
    disabled.update(self.model)
    self.assertEmpty(disabled.shadow)
    for ema in (disabled, ema_lib.ParameterEMA(self.model)):
      self._fill(5.0)  # E.g. weights loaded after construction.
      with ema.average_parameters(self.model):
        self._assert_params(5.0, 5.0)

  def test_warmup(self):
    # warmup=1: running mean (decays 0, 1/2, 2/3) until 3/4 exceeds the decay,
    # i.e. 0.7 * mean(1, 2, 3) + 0.3 * 4.
    ema = ema_lib.ParameterEMA(self.model, decay=0.7, warmup=1.0)
    for value in (1.0, 2.0, 3.0, 4.0):
      self._fill(value)
      ema.update(self.model)
    torch.testing.assert_close(ema.shadow[0], torch.tensor([[2.6]]))
    # warmup=0: copy, then the decay from the second update on.
    ema = ema_lib.ParameterEMA(self.model, decay=0.7, warmup=0.0)
    for value in (1.0, 2.0):
      self._fill(value)
      ema.update(self.model)
    torch.testing.assert_close(ema.shadow[0], torch.tensor([[1.3]]))

  def test_invalid_decay(self):
    with self.assertRaises(ValueError):
      ema_lib.ParameterEMA(self.model, decay=1.0)
    with self.assertRaises(ValueError):
      ema_lib.ParameterEMA(self.model, warmup=-1.0)


if __name__ == "__main__":
  absltest.main()
