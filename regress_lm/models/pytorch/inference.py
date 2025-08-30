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

"""PyTorch implementations of inference functions."""

from typing import Sequence

from regress_lm import core
from regress_lm.models import base
import torch


Tensor = torch.Tensor


class RAFTInferenceFn(base.InferenceFn[Tensor]):
  """Performs RAFT inference to collect a single float.

  The mean can be estimated via sum_{y in Grid}[y * prob(y|x)].

  NOTE: `infer` logic can also be used in training with a MSE loss.
  """

  def __init__(self, model: base.Model[Tensor], grid_points: Sequence[float]):
    self.model = model
    self.grid_points = grid_points

  def infer(self, inputs: Sequence[core.ExampleInput]) -> Tensor:
    # Create one large batch for all inputs and all grid points
    all_grid_examples = []
    for example_input in inputs:
      for y_val in self.grid_points:
        all_grid_examples.append(core.Example(x=example_input.x, y=y_val))

    # Convert and run the model on the entire batch in a single forward pass
    # This batch will have a size of (L_inputs * L_grid_points)
    grid_batch = self.model.convert_examples(all_grid_examples)
    log_probs = self.model.log_prob(grid_batch)

    probs = torch.exp(log_probs)
    probs_reshaped = probs.view(len(inputs), len(self.grid_points))

    # Ensure grid_points is a tensor on the correct device
    grid_points_tensor = torch.tensor(
        self.grid_points, device=probs.device, dtype=probs.dtype
    )

    # Compute the weighted average for each input using matrix multiplication
    # This is equivalent to a batched dot product
    return torch.matmul(probs_reshaped, grid_points_tensor)
