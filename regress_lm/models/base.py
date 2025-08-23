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

"""Base class for a RegressLM model."""

import abc
from typing import Generic, Sequence, TypeVar
import numpy as np
from regress_lm import core

# Low-level tensor type.
TensorT = TypeVar('TensorT')


class Model(Generic[TensorT], abc.ABC):
  """Abstract class for a Model.

  Uses generic types to allow different low-level tensor packages (Jax, PyTorch,
  etc.). ExampleT can be jax.Array, torch.Tensor, etc.

  Conversion between high-level and low-level implementations can be done via
  wrappers.
  """

  @abc.abstractmethod
  def compute_losses_and_metrics(
      self, examples: dict[str, TensorT]
  ) -> tuple[TensorT, dict[str, TensorT]]:
    """Computes per example loss and aggregate metrics for given examples."""

  @abc.abstractmethod
  def decode(
      self, inputs: dict[str, TensorT], num_samples: int
  ) -> tuple[TensorT, np.ndarray]:
    """Decodes tokens and returns them and corresponding floats."""

  @abc.abstractmethod
  def log_prob(self, examples: dict[str, TensorT]) -> TensorT:
    """Returns log probability of y given x."""

  @abc.abstractmethod
  def convert_inputs(
      self, inputs: Sequence[core.ExampleInput]
  ) -> dict[str, TensorT]:
    """Converts high-level inputs to batched low-level inputs."""

  @abc.abstractmethod
  def convert_examples(
      self, examples: Sequence[core.Example]
  ) -> dict[str, TensorT]:
    """Converts high-level examples to batched low-level examples."""


class FineTuner(abc.ABC):

  @abc.abstractmethod
  def fine_tune(
      self,
      examples: Sequence[core.Example],
      validation_examples: Sequence[core.Example] | None = None,
      seed: int | None = None,
  ) -> None:
    """Fine-tunes the model on the given examples.

    For full compatibility, assumes the model is stateful and changes itself
    (hence the return being None).

    Args:
      examples: Training examples.
      validation_examples: Validation examples for early-stopping. If None, uses
        training examples.
      seed: Random seed for fine-tuning.

    Returns:
      None
    """
