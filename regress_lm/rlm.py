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

import functools
from typing import Sequence
import numpy as np
from regress_lm import core
from regress_lm import tokenizers
from regress_lm import vocabs


class RegressLM:
  """User-facing API for RegressLM."""

  def __init__(self, model: core.Model):
    self.model = model

  def fine_tune(
      self,
      examples: Sequence[core.Example],
      validation_examples: Sequence[core.Example] | None = None,
      seed: int | None = None,
      **kwargs,
  ) -> None:
    """Fine-tunes the model against the provided examples."""

    # pylint: disable=g-import-not-at-top
    from regress_lm.pytorch import fine_tuning as pytorch_fine_tuning
    from regress_lm.pytorch import optimizers

    # NOTE: This adds extra GPU memory usage, so we don't add into init.
    fine_tuner = pytorch_fine_tuning.PyTorchFineTuner(
        self.model,
        optimizer_factory=kwargs.get("optimizer_factory", None)
        or functools.partial(optimizers.muon_adamw, lr=1e-4),
        max_epochs=kwargs.get("max_epochs", 100),
        batch_size=kwargs.get("batch_size", None),
        batch_size_per_device=kwargs.get("batch_size_per_device", None),
        patience=kwargs.get("patience", 1),
        use_lora=kwargs.get("use_lora", False),
        lora_r=kwargs.get("lora_r", 8),
        lora_alpha=kwargs.get("lora_alpha", 16),
        lora_dropout=kwargs.get("lora_dropout", 0.0),
        target_modules=kwargs.get(
            "target_modules", pytorch_fine_tuning.DEFAULT_LORA_TARGET_MODULES
        ),
    )
    fine_tuner.fine_tune(examples, validation_examples, seed)

  @classmethod
  def from_scratch(cls, **kwargs) -> "RegressLM":
    """Creates a RegressLM with default model and finetuner."""

    # pylint: disable=g-import-not-at-top
    from regress_lm.pytorch import model as pytorch_model
    from regress_lm.pytorch import encoders

    architecture_kwargs = dict(
        d_model=kwargs.get("d_model", 512),
        num_encoder_layers=kwargs.get("num_encoder_layers", 2),
        num_decoder_layers=kwargs.get("num_decoder_layers", 2),
        decoder_dropout=kwargs.get("decoder_dropout", 0.0),
        encoder_type=encoders.EncoderType.VANILLA,
        additional_encoder_kwargs=kwargs.get("additional_encoder_kwargs", {}),
    )

    config = pytorch_model.PyTorchModelConfig(
        encoder_vocab=kwargs.get("encoder_vocab")
        or vocabs.SentencePieceVocab.from_t5(),
        decoder_vocab=kwargs.get("decoder_vocab")
        or vocabs.DecoderVocab(tokenizers.IEEEFloatTokenizer()),
        max_input_len=kwargs.get("max_input_len", 2048),
        max_num_objs=kwargs.get("max_num_objs", 1),
        z_loss_coef=kwargs.get("z_loss_coef", None),
        architecture_kwargs=architecture_kwargs,
    )

    return cls(model=config.make_model())

  @classmethod
  def from_t5gemma_encoder(
      cls,
      model_name: str = "google/t5gemma-s-s-prefixlm",
      freeze_encoder: bool = False,
      random_init: bool = False,
      **kwargs,
  ) -> "RegressLM":
    """T5Gemma (v1 or 2) encoder w/ custom decoder."""

    # pylint: disable=g-import-not-at-top
    from regress_lm.pytorch import model as pytorch_model
    from regress_lm.pytorch import encoders

    architecture_kwargs = dict(
        d_model=kwargs.get("d_model", 512),
        num_encoder_layers=0,  # Dummy value, will be ignored.
        num_decoder_layers=kwargs.get("num_decoder_layers", 2),
        decoder_dropout=kwargs.get("decoder_dropout", 0.0),
        encoder_type=encoders.EncoderType.T5GEMMA,
        additional_encoder_kwargs={
            "model_name": model_name,
            "freeze_weights": freeze_encoder,
            "random_init": random_init,
            "dropout": kwargs.get("dropout", 0.0),
            "use_grad_ckpt": kwargs.get("use_grad_ckpt", False),
        },
    )

    config = pytorch_model.PyTorchModelConfig(
        encoder_vocab=kwargs.get("encoder_vocab")
        or vocabs.HuggingFaceVocab(model_name),
        decoder_vocab=kwargs.get("decoder_vocab")
        or vocabs.DecoderVocab(tokenizers.IEEEFloatTokenizer()),
        max_input_len=kwargs.get("max_input_len", 2048),
        max_num_objs=kwargs.get("max_num_objs", 1),
        z_loss_coef=kwargs.get("z_loss_coef", None),
        architecture_kwargs=architecture_kwargs,
    )

    return cls(model=config.make_model())

  def sample(
      self, xs: Sequence[core.ExampleInput], num_samples: int
  ) -> Sequence[np.ndarray]:
    """Samples from the model."""
    examples = self.model.converter.convert_inputs(xs)
    _, output_floats = self.model.decode(examples, num_samples)
    return [y.squeeze(axis=0) for y in np.split(output_floats, len(xs), axis=0)]
