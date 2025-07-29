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

"""T5Gemma wrapper."""

from typing import Any, Callable, Sequence

import numpy as np
from regress_lm import core
from regress_lm.models import base as model_base
import torch
from torch import nn
import torch.nn.functional as F
import transformers

Tensor = torch.Tensor

IGNORE_TOKEN_ID = -100  # Default ignore_index in torch `cross_entropy`.
# Also hardcoded in t5gemma for shift_right. We MUST use it.


def default_y_to_str_fn(ys: float | Sequence[float], precision: int = 4) -> str:
  """Default function to convert a float or sequence of floats to a string."""
  format_fn = lambda y: format(y, f'.{precision}e')
  if isinstance(ys, float):
    return format_fn(ys)
  if isinstance(ys, Sequence):
    return ','.join(format_fn(y) for y in ys)
  raise ValueError(f'Unsupported type: {type(ys)}')


class T5GemmaModel(nn.Module, model_base.Model[Tensor]):
  """Comment."""

  def __init__(
      self,
      model_name: str,
      max_input_len: int = 2048,
      max_decode_len: int = 32,  # Max tokens for the numeric output string
      x_to_str_fn: Callable[[str], str] = lambda x: x,  # For prompting.
      y_to_str_fn: Callable[[float | Sequence[float]], str] = default_y_to_str_fn,  # pylint: disable=line-too-long
      model_kwargs: dict[str, Any] | None = None,  # For T5Gemma kwargs.
      tokenizer_kwargs: dict[str, Any] | None = None,  # For tokenizer kwargs.
  ):
    super().__init__()
    self.model_name = model_name
    self.max_input_len = max_input_len
    self.max_decode_len = max_decode_len

    self.x_to_str_fn = x_to_str_fn
    self.y_to_str_fn = y_to_str_fn

    self.model = transformers.T5GemmaForConditionalGeneration.from_pretrained(
        model_name, **(model_kwargs or {})
    )
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, **(tokenizer_kwargs or {})
    )

  def compute_loss_and_metrics(
      self, examples: dict[str, Tensor]
  ) -> tuple[Tensor, dict[str, Tensor]]:
    outputs = self.model(
        input_ids=examples['input_ids'],
        attention_mask=examples['attention_mask'],
        labels=examples['labels'],
    )
    loss = outputs.loss
    return loss, {'loss': loss.detach()}

  @torch.no_grad()
  def decode(
      self, inputs: dict[str, Tensor], num_samples: int
  ) -> tuple[Tensor, np.ndarray]:
    """Decodes strings that may or may not be valid floats.

    To allow returning a single np.ndarray, we instead return a string tensor,
    and allow users to parse.

    Args:
      inputs: Converted inputs.
      num_samples: The number of samples to generate.

    Returns:
      A tuple of (decoded_tokens, output_strings).
      decoded_tokens: Decoded tokens. Shape (B, S, L_decode).
      output_strings: A numpy array of generated strings of shape (B, S).
    """
    self.model.eval()
    batch_size = inputs['input_ids'].shape[0]

    # Contains both old and new tokens.
    decoded_tokens = self.model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=self.max_decode_len,
        num_return_sequences=num_samples,
        do_sample=True,
        pad_token_id=self.tokenizer.pad_token_id,
    )
    decoded_texts = self.tokenizer.batch_decode(
        decoded_tokens, skip_special_tokens=True
    )

    # Remove the starter token and reshape. (B, S, L_decode)
    decoded_tokens = decoded_tokens[:, 1:].view(
        batch_size, num_samples, self.max_decode_len
    )
    output_strings = np.array(decoded_texts).reshape(batch_size, num_samples)
    return decoded_tokens, output_strings

  def log_prob(self, examples: dict[str, Tensor]) -> Tensor:
    self.model.eval()
    outputs = self.model(
        input_ids=examples['input_ids'],
        attention_mask=examples['attention_mask'],
        labels=examples['labels'],
    )
    labels = examples['labels']

    log_sm = F.log_softmax(outputs.logits, dim=-1)  # (B, L, V)
    log_probs = torch.gather(log_sm, 2, labels.unsqueeze(-1))  # (B, L, 1)
    log_probs = log_probs.squeeze(-1)  # (B, L)

    mask = (labels != IGNORE_TOKEN_ID).float()  # (B, L)
    return (log_probs * mask).sum(dim=1)  # (B,)

  def convert_inputs(
      self, inputs: Sequence[core.ExampleInput]
  ) -> dict[str, Tensor]:
    """Converts string inputs to tokenized tensors for the encoder."""
    return self.tokenizer(
        [self.x_to_str_fn(ex.x) for ex in inputs],
        padding='max_length',
        truncation=True,
        max_length=self.max_input_len,
        return_tensors='pt',
    )

  def convert_examples(
      self, examples: Sequence[core.Example]
  ) -> dict[str, Tensor]:
    # Tokenize the target strings to create the labels
    output = self.tokenizer(
        [self.y_to_str_fn(ex.y) for ex in examples],
        padding='max_length',
        truncation=True,
        max_length=self.max_decode_len,
        return_tensors='pt',
    )
    labels = output.input_ids

    labels[labels == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    return {'labels': labels, **self.convert_inputs(examples)}
