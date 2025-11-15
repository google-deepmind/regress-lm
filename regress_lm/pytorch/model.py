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

"""PyTorch implementation of a RegressLM."""

# pytype:disable=attribute-error
from concurrent import futures
import dataclasses
import functools
from typing import Any, Sequence
import numpy as np
from regress_lm import core
from regress_lm import vocabs
from regress_lm.pytorch import architecture
import torch
from torch import nn
import torch.nn.functional as F

NEG_INF = -1.0e7

# Dict Keys: "encoder_input", "decoder_input", "decoder_target"
Tensor = torch.Tensor


@dataclasses.dataclass(frozen=True)
class PyTorchModelConfig:
  """Configuration for creating a RegressLM model and its data converter."""

  # Vocabularies
  encoder_vocab: vocabs.EncoderVocab[str]
  decoder_vocab: vocabs.DecoderVocab[float]

  # Core Dimensions
  max_input_len: int
  architecture_kwargs: dict[str, Any]
  max_num_objs: int = 1

  # Other settings
  z_loss_coef: float | None = None

  @property
  def decode_len(self) -> int:
    return self.max_num_objs * self.decoder_vocab.num_tokens_per_obj

  def make_converter(self) -> 'PyTorchConverter':
    """Factory method to create a data converter from this config."""
    return PyTorchConverter(config=self)

  def make_model(self, compile_model: bool = True) -> 'PyTorchModel':
    """Factory method to create a model from this config."""
    return PyTorchModel(config=self, compile_model=compile_model)


class PyTorchConverter(core.Converter[Tensor]):
  """Converts high-level inputs and examples to batched low-level inputs."""

  def __init__(self, config: PyTorchModelConfig):
    self.cfg = config

  def convert_inputs(
      self, inputs: Sequence[core.ExampleInput]
  ) -> dict[str, Tensor]:
    strings = [example.x for example in inputs]
    encoder_token_ids = [
        self.cfg.encoder_vocab.to_token_ids(s) for s in strings
    ]
    encoder_input = [self._pad_or_truncate(t) for t in encoder_token_ids]
    return {'encoder_input': torch.tensor(encoder_input)}

  def convert_examples(
      self, examples: Sequence[core.Example]
  ) -> dict[str, Tensor]:
    y_values = [example.y for example in examples]
    y_tokens_list = [self.cfg.decoder_vocab.to_token_ids(y) for y in y_values]

    decoder_inputs = []
    decoder_targets = []
    pad_id = self.cfg.decoder_vocab.bos_pad_id

    for t in y_tokens_list:
      padding_needed = self.cfg.decode_len - len(t)
      # Input: [pad, t_1, ..., t_n, pad, ..., pad]
      decoder_inputs.append([pad_id] + t + [pad_id] * padding_needed)
      # Target: [t_1, ..., t_n, pad, ..., pad]
      decoder_targets.append(t + [pad_id] * (padding_needed + 1))

    decoder_out = {
        'decoder_input': torch.tensor(decoder_inputs),
        'decoder_target': torch.tensor(decoder_targets),
    }
    return self.convert_inputs(examples) | decoder_out

  def _pad_or_truncate(self, token_ids: list[int]) -> list[int]:
    encoder_pad_idx = self.cfg.encoder_vocab.pad_id
    if len(token_ids) > self.cfg.max_input_len:
      return token_ids[: self.cfg.max_input_len]
    return token_ids + [encoder_pad_idx] * (
        self.cfg.max_input_len - len(token_ids)
    )


class PyTorchModel(nn.Module, core.Model[Tensor]):
  """PyTorch implementation of a RegressLM.

  NOTE: External callers are responsible for train() and eval() modes which deal
  with batchnorm and dropout.
  """

  def __init__(self, config: PyTorchModelConfig, compile_model: bool):
    super().__init__()
    self.cfg = config

    self.encoder_decoder = architecture.EncoderDecoder(
        encoder_vocab_size=len(self.cfg.encoder_vocab),
        decoder_vocab_size=len(self.cfg.decoder_vocab),
        encoder_pad_idx=self.cfg.encoder_vocab.pad_id,
        max_encoder_len=self.cfg.max_input_len,
        max_decoder_len=self.cfg.decode_len + 1,
        **self.cfg.architecture_kwargs,
    )

    if compile_model:
      self.encoder_decoder = torch.compile(self.encoder_decoder)

  @property
  def device(self) -> torch.device:
    return next(self.parameters()).device

  def to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v.to(self.device) for k, v in batch.items()}

  def compute_losses_and_metrics(
      self, examples: dict[str, Tensor]
  ) -> tuple[Tensor, dict[str, Tensor]]:
    examples = self.to_device(examples)
    metrics = {}
    logits = self.encoder_decoder.forward(
        examples['encoder_input'], examples['decoder_input']
    )
    targets = examples['decoder_target']

    ce_loss_per_tokens = F.cross_entropy(
        logits.permute(0, 2, 1),
        targets,
        ignore_index=self.cfg.decoder_vocab.bos_pad_id,
        reduction='none',
    )

    # Average loss per non-padded token.
    loss_mask = (targets != self.cfg.decoder_vocab.bos_pad_id).float()
    num_tokens = loss_mask.sum(dim=1).clamp(min=1)
    loss_per_example = ce_loss_per_tokens.sum(dim=1) / num_tokens

    if self.cfg.z_loss_coef is not None:
      log_z = torch.logsumexp(logits, dim=-1)
      z_loss_per_token = self.cfg.z_loss_coef * (log_z**2)
      loss_mask = (targets != self.cfg.decoder_vocab.bos_pad_id).float()
      z_loss_per_example = (z_loss_per_token * loss_mask).sum(dim=1)
      loss_per_example = loss_per_example + z_loss_per_example

    metrics['loss_mean'] = loss_per_example.mean().detach()
    return loss_per_example, metrics

  @torch.no_grad()
  def decode(
      self,
      inputs: dict[str, Tensor],
      num_samples: int,
      temperature: float = 1.0,
  ) -> tuple[Tensor, np.ndarray]:
    inputs = self.to_device(inputs)

    encoder_input = inputs['encoder_input']  # (B, L_src)
    batch_size = encoder_input.shape[0]
    expanded_batch_size = batch_size * num_samples
    # memory: (B, L_src, D_model), memory_key_padding_mask: (B, L_src)
    memory, memory_key_padding_mask = self.encoder_decoder.encode(encoder_input)

    # Expand encoder outputs and masks for num_samples
    # Effectively, new batch_size = B * num_samples
    # memory: (B, L_src, D) -> (B, 1, L_src, D) -> (B, S, L_src, D)
    # -> (B*S, L_src, D)
    expanded_memory = (
        memory.unsqueeze(1)
        .expand(-1, num_samples, -1, -1)
        .reshape(batch_size * num_samples, memory.size(1), memory.size(2))
    )
    expanded_memory_key_padding_mask = (
        memory_key_padding_mask.unsqueeze(1)
        .expand(-1, num_samples, -1)
        .reshape(batch_size * num_samples, memory_key_padding_mask.size(1))
    )

    # Initialize decoder input for the expanded batch, start with <pad>.
    current_tgt_ids = torch.full(
        (expanded_batch_size, 1),
        self.cfg.decoder_vocab.bos_pad_id,
        dtype=torch.long,
        device=self.device,
    )

    # Store all generated token IDs for all sequences in the expanded batch
    generated_sequences_ids = torch.zeros(
        (expanded_batch_size, self.cfg.decode_len),
        dtype=torch.long,
        device=self.device,
    )

    for step_idx in range(self.cfg.decode_len):
      ids_step = generated_sequences_ids[:, :step_idx].to('cpu')

      def get_allowed_tokens(i: int) -> list[int]:
        prev_tokens = ids_step[i].tolist()  # pylint: disable=cell-var-from-loop
        return self.cfg.decoder_vocab.possible_next_token_ids(prev_tokens)

      with futures.ThreadPoolExecutor() as executor:
        results = executor.map(get_allowed_tokens, range(expanded_batch_size))

      curr_mask = torch.zeros(
          (expanded_batch_size, len(self.cfg.decoder_vocab)),
          dtype=torch.float32,
          device='cpu',
      )
      for i, allowed_inds in enumerate(results):
        if allowed_inds:
          curr_mask[i, allowed_inds] = 1.0

      curr_mask = curr_mask.to(self.device)

      # Get logits for the next token for all (B * num_samples) sequences
      # Shape: (B*S, V)
      logits = self.encoder_decoder.next_token_logits(
          current_tgt_ids, expanded_memory, expanded_memory_key_padding_mask
      )
      masked_logits = (1.0 - curr_mask) * NEG_INF + curr_mask * logits

      # Apply temperature sampling, 1 token for each of the B*S sequences
      probs = F.softmax(masked_logits / temperature, dim=-1)
      token_ids = torch.multinomial(probs, num_samples=1)  # (B*S, 1)
      # Store the predicted token IDs
      generated_sequences_ids[:, step_idx] = token_ids.squeeze(-1)

      # Prepare input for the next step, but only if not the last float token
      if step_idx < self.cfg.decode_len - 1:
        current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

    # Reshape outputs back to (B, num_samples, L_decode)
    final_decoded_ids = generated_sequences_ids.view(
        batch_size, num_samples, self.cfg.decode_len
    )
    final_decoded_ids = final_decoded_ids.to('cpu')  # Don't hog the GPU.

    # Compute equivalent floats. Object type is used in case of special values.
    output_floats = np.empty(
        (batch_size, num_samples, self.cfg.max_num_objs), dtype=object
    )
    for b in range(batch_size):
      for s_idx in range(num_samples):
        output_floats[b, s_idx, :] = self.cfg.decoder_vocab.from_token_ids(
            final_decoded_ids[b, s_idx, :].tolist()
        )

    return final_decoded_ids, output_floats

  def log_prob(self, examples: dict[str, Tensor]) -> Tensor:
    examples = self.to_device(examples)
    enc_input = examples['encoder_input']
    dec_input = examples['decoder_input']
    dec_target = examples['decoder_target']

    logits = self.encoder_decoder.forward(enc_input, dec_input)
    log_probs = F.log_softmax(logits, dim=-1)

    true_log_probs = torch.gather(
        log_probs, dim=2, index=dec_target.unsqueeze(-1)
    )

    pad_mask = dec_target != self.cfg.decoder_vocab.bos_pad_id
    true_log_probs_masked = true_log_probs.squeeze(-1) * pad_mask
    sequence_sum_log_probs = true_log_probs_masked.sum(dim=1)
    return sequence_sum_log_probs

  @functools.cached_property
  def converter(self) -> core.Converter[Tensor]:
    return self.cfg.make_converter()
