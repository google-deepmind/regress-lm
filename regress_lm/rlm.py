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

"""Very high-level class for users."""

from typing import Sequence
import numpy as np
from regress_lm import core
from regress_lm import tokenizers
from regress_lm import vocabs
from regress_lm.models import base as model_base
from regress_lm.models.pytorch import model as pytorch_model


class RegressLM:
  """User-facing API for RegressLM."""

  def __init__(self, model: model_base.Model, fine_tuner: model_base.FineTuner):
    self.model = model
    self.fine_tuner = fine_tuner

  def fine_tune(
      self,
      examples: Sequence[core.Example],
      validation_examples: Sequence[core.Example] | None = None,
      **kwargs,
  ):
    self.fine_tuner.fine_tune(examples, validation_examples, **kwargs)

  @classmethod
  def from_default(cls, **kwargs) -> "RegressLM":
    """Creates a RegressLM with default model and finetuner."""

    encoder_vocab = vocabs.SentencePieceVocab.from_t5()
    decoder_vocab = vocabs.DecoderVocab(tokenizers.P10Tokenizer())
    model = pytorch_model.PyTorchModel(
        encoder_vocab=encoder_vocab,
        decoder_vocab=decoder_vocab,
        max_input_len=kwargs.get("max_input_len", 2048),
        max_num_objs=kwargs.get("max_num_objs", 1),
        learning_rate=kwargs.get("learning_rate", 1e-4),
        d_model=kwargs.get("d_model", 512),
        nhead=kwargs.get("nhead", 8),
        num_encoder_layers=kwargs.get("num_encoder_layers", 2),
        num_decoder_layers=kwargs.get("num_decoder_layers", 2),
        dim_feedforward=kwargs.get("dim_feedforward", 2048),
        dropout=kwargs.get("dropout", 0.0),
    )

    fine_tuner = pytorch_model.PyTorchFineTuner(model)
    return cls(model, fine_tuner)

  def sample(
      self, xs: Sequence[core.ExampleInput], num_samples: int
  ) -> Sequence[np.ndarray]:
    """Samples from the model."""
    examples = self.model.convert_inputs(xs)
    _, output_floats = self.model.decode(examples, num_samples)
    return [y.squeeze(axis=0) for y in np.split(output_floats, len(xs), axis=0)]
