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


class RegressLM:
  """User-facing API for RegressLM."""

  def __init__(self, model: core.Model, fine_tuner: core.FineTuner):
    self.model = model
    self.fine_tuner = fine_tuner

  def fine_tune(
      self,
      examples: Sequence[core.Example],
      validation_examples: Sequence[core.Example] | None = None,
      seed: int | None = None,
  ):
    self.fine_tuner.fine_tune(examples, validation_examples, seed=seed)

  @classmethod
  def from_scratch(cls, device: str | None = None, **kwargs) -> "RegressLM":
    """Creates a RegressLM with default model and finetuner."""
    # pylint: disable=g-import-not-at-top
    import torch
    from torch import optim
    from regress_lm.pytorch import model as pytorch_model
    from regress_lm.pytorch import fine_tuning as pytorch_fine_tuning
    from regress_lm.pytorch import encoders

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = pytorch_model.PyTorchModel(
        encoder_vocab=kwargs.get("encoder_vocab")
        or vocabs.SentencePieceVocab.from_t5(),
        decoder_vocab=kwargs.get("decoder_vocab")
        or vocabs.DecoderVocab(tokenizers.P10Tokenizer()),
        max_input_len=kwargs.get("max_input_len", 2048),
        max_num_objs=kwargs.get("max_num_objs", 1),
        compile_model=kwargs.get("compile_model", True),
        d_model=kwargs.get("d_model", 512),
        num_encoder_layers=kwargs.get("num_encoder_layers", 2),
        num_decoder_layers=kwargs.get("num_decoder_layers", 2),
        encoder_type=kwargs.get("encoder_type", encoders.EncoderType.VANILLA),
        additional_encoder_kwargs=kwargs.get("additional_encoder_kwargs", {}),
    ).to(device)

    optimizer = kwargs.get("optimizer", None) or optim.Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    fine_tuner = pytorch_fine_tuning.PyTorchFineTuner(
        model,
        optimizer=optimizer,
        max_epochs=kwargs.get("max_epochs", 100),
        batch_size=kwargs.get("batch_size", None),
        batch_size_per_device=kwargs.get("batch_size_per_device", None),
        patience=kwargs.get("patience", 1),
    )
    return cls(model, fine_tuner)

  @classmethod
  def from_t5gemma_encoder(
      cls,
      model_name: str = "google/t5gemma-s-s-prefixlm",
      device: str | None = None,
      **kwargs
  ) -> "RegressLM":
    """Frozen T5Gemma encoder w/ custom decoder."""
    # pylint: disable=g-import-not-at-top
    import torch
    from torch import optim
    from regress_lm.pytorch import model as pytorch_model
    from regress_lm.pytorch import fine_tuning as pytorch_fine_tuning
    from regress_lm.pytorch import encoders

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = pytorch_model.PyTorchModel(
        encoder_vocab=vocabs.HuggingFaceVocab(model_name),
        decoder_vocab=kwargs.get("decoder_vocab")
        or vocabs.DecoderVocab(tokenizers.P10Tokenizer()),
        max_input_len=kwargs.get("max_input_len", 2048),
        max_num_objs=kwargs.get("max_num_objs", 1),
        compile_model=kwargs.get("compile_model", True),
        d_model=kwargs.get("d_model", 512),
        num_encoder_layers=0,  # Dummy value, will be ignored.
        num_decoder_layers=kwargs.get("num_decoder_layers", 2),
        encoder_type=kwargs.get("encoder_type", encoders.EncoderType.T5GEMMA),
        additional_encoder_kwargs=kwargs.get("additional_encoder_kwargs", {}),
    ).to(device)

    optimizer = kwargs.get("optimizer", None) or optim.Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    fine_tuner = pytorch_fine_tuning.PyTorchFineTuner(
        model,
        optimizer=optimizer,
        max_epochs=kwargs.get("max_epochs", 100),
        batch_size=kwargs.get("batch_size", None),
        batch_size_per_device=kwargs.get("batch_size_per_device", None),
        patience=kwargs.get("patience", 1),
    )
    return cls(model, fine_tuner)

  def sample(
      self, xs: Sequence[core.ExampleInput], num_samples: int
  ) -> Sequence[np.ndarray]:
    """Samples from the model."""
    examples = self.model.convert_inputs(xs)
    _, output_floats = self.model.decode(examples, num_samples)
    return [y.squeeze(axis=0) for y in np.split(output_floats, len(xs), axis=0)]
