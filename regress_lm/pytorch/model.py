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
  num_workers: int = 16  # Multithreaded tokenization + constrained decoding.

  @property
  def decode_len(self) -> int:
    return self.max_num_objs * self.decoder_vocab.num_tokens_per_obj

  def make_converter(self) -> 'PyTorchConverter':
    """Factory method to create a data converter from this config."""
    return PyTorchConverter(config=self)

  def make_model(self) -> 'PyTorchModel':
    """Factory method to create a model from this config."""
    return PyTorchModel(self)


ThreadPoolExecutor = futures.ThreadPoolExecutor


class PyTorchConverter(core.Converter[Tensor]):
  """Converts high-level inputs and examples to batched low-level inputs."""

  def __init__(self, config: PyTorchModelConfig):
    self.cfg = config

  def convert_inputs(
      self, inputs: Sequence[core.ExampleInput]
  ) -> dict[str, Tensor]:
    strings = [example.x for example in inputs]

    with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as executor:
      encoder_token_ids = list(
          executor.map(self.cfg.encoder_vocab.to_token_ids, strings)
      )

    batch_size, max_len = len(strings), self.cfg.max_input_len

    # Pre-allocate padded arrays for efficiency + inject in-place.
    encoder_input_np = np.full(
        (batch_size, max_len), self.cfg.encoder_vocab.pad_id, dtype=np.int64
    )
    input_token_lens_np = np.zeros(batch_size, dtype=np.int64)
    for i, tokens in enumerate(encoder_token_ids):
      tok_len = len(tokens)
      input_token_lens_np[i] = tok_len

      # Truncate if necessary and insert
      insert_len = min(tok_len, max_len)
      encoder_input_np[i, :insert_len] = tokens[:insert_len]

    return {
        'encoder_input': torch.from_numpy(encoder_input_np),
        'input_token_lens': torch.from_numpy(input_token_lens_np),
    }

  def convert_examples(
      self, examples: Sequence[core.Example]
  ) -> dict[str, Tensor]:
    y_values = [example.y for example in examples]

    with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as executor:
      y_tokens_list = list(
          executor.map(self.cfg.decoder_vocab.to_token_ids, y_values)
      )

    batch_size, dec_len = len(y_values), self.cfg.decode_len
    pad_id = self.cfg.decoder_vocab.bos_pad_id

    # Pre-allocate padded arrays for efficiency + inject in-place.
    dec_input = np.full((batch_size, dec_len + 1), pad_id, dtype=np.int64)
    dec_target = np.full((batch_size, dec_len + 1), pad_id, dtype=np.int64)
    for i, tokens in enumerate(y_tokens_list):
      insert_len = min(len(tokens), dec_len)
      # Input: [pad, t_1, ..., t_n, pad, ...]
      dec_input[i, 1 : insert_len + 1] = tokens[:insert_len]
      # Target: [t_1, ..., t_n, pad, ...]
      dec_target[i, :insert_len] = tokens[:insert_len]

    decoder_out = {
        'decoder_input': torch.from_numpy(dec_input),
        'decoder_target': torch.from_numpy(dec_target),
    }
    return self.convert_inputs(examples) | decoder_out


class PyTorchModel(nn.Module, core.Model[Tensor]):
  """PyTorch implementation of a RegressLM.

  NOTE: External callers are responsible for train() and eval() modes which deal
  with batchnorm and dropout.
  """

  def __init__(self, config: PyTorchModelConfig):
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

  @property
  def device(self) -> torch.device:
    return next(self.parameters()).device

  def to_device(
      self, batch: dict[str, Tensor] | Tensor
  ) -> dict[str, Tensor] | Tensor:
    if isinstance(batch, dict):
      return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
    return batch.to(self.device, non_blocking=True)

  def compute_losses_and_metrics(
      self, examples: dict[str, Tensor]
  ) -> tuple[Tensor, dict[str, Tensor]]:
    metrics = {}
    logits = self.encoder_decoder.forward(
        self.to_device(examples['encoder_input']),
        self.to_device(examples['decoder_input']),
    )
    targets = self.to_device(examples['decoder_target'])

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
    metrics['max_input_token_len'] = examples['input_token_lens'].max().detach()
    return loss_per_example, metrics

  @torch.no_grad()
  def decode(
      self,
      inputs: dict[str, Tensor],
      num_samples: int,
      temperature: float = 1.0,
  ) -> tuple[Tensor, np.ndarray]:
    encoder_input = self.to_device(inputs['encoder_input'])  # (B, L_src)
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
    with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as executor:
      for step_idx in range(self.cfg.decode_len):
        ids_step = generated_sequences_ids[:, :step_idx].to('cpu')

        def get_allowed_tokens(i: int) -> list[int]:
          prev_tokens = ids_step[i].tolist()  # pylint: disable=cell-var-from-loop
          return self.cfg.decoder_vocab.possible_next_token_ids(prev_tokens)

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
    logits = self.encoder_decoder.forward(
        self.to_device(examples['encoder_input']),
        self.to_device(examples['decoder_input']),
    )
    log_probs = F.log_softmax(logits, dim=-1)

    dec_target = self.to_device(examples['decoder_target'])
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
