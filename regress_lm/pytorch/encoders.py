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

"""Different encoder types."""

import abc
import copy
import enum
import functools
import math
from typing import Callable, Literal
from absl import logging
# The following import is needed for loading T5Gemma models.
import safetensors.torch  # pylint: disable=unused-import
import torch
from torch import nn
from torch.nn import functional as F
import transformers

# pylint: disable=g-import-not-at-top
try:
  import mamba_ssm  # pytype: disable=import-error
except ImportError:
  mamba_ssm = None


class BaseEncoder(nn.Module, abc.ABC):
  """Abstract base class for all encoder models."""

  @property
  @abc.abstractmethod
  def hidden_dim(self) -> int:
    """Hidden dimension of the encoder output, needed for decoder."""
    raise NotImplementedError("Subclasses must implement 'hidden_dim'.")

  @abc.abstractmethod
  def forward(
      self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor | None
  ) -> torch.Tensor:
    """Encodes the source sequence.

    Args:
        src_ids: The input tensor of token ids.
        src_key_padding_mask: The padding mask for the source tensor.

    Returns:
        The encoded memory tensor.
    """
    raise NotImplementedError("Subclasses must implement the 'forward' method.")


class _RotaryPositionalEmbedding(nn.Module):
  """Rotary Positional Embedding (RoPE)."""

  def __init__(self, d_model: int, max_len: int, base: int = 10000):
    super().__init__()
    # Note: d_model is the head dimension in our case.
    inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
    self.register_buffer("inv_freq", inv_freq)

    t = torch.arange(max_len, dtype=self.inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    # freqs shape is (max_len, d_model / 2)
    # The cached sin/cos tables should have a feature dimension of d_model / 2
    self.register_buffer(
        "cos_cached", freqs.cos()[None, None, :, :], persistent=False
    )
    self.register_buffer(
        "sin_cached", freqs.sin()[None, None, :, :], persistent=False
    )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = x.shape[2]  # x has shape (B, n_heads, L, head_dim).
    # Return tensors of shape (1, 1, L, head_dim / 2)
    return (
        self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
    )


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Applies RoPE to query and key tensors."""
  # q, k have shape (B, H, L, D_h)
  # cos, sin have shape (B, H, L, D_h / 2) after broadcasting

  # Reshape q and k to separate even and odd dimensions
  # TODO: Consider using torch.repeat_interleave.
  q_reshaped = q.float().reshape(*q.shape[:-1], -1, 2)
  k_reshaped = k.float().reshape(*k.shape[:-1], -1, 2)
  q_even, q_odd = q_reshaped[..., 0], q_reshaped[..., 1]
  k_even, k_odd = k_reshaped[..., 0], k_reshaped[..., 1]
  # q_even, q_odd, k_even, k_odd have shape (B, H, L, D_h / 2)

  # Apply rotation. All tensors in this operation have a final dim of D_h / 2.
  q_out = torch.stack(
      [q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], -1
  ).flatten(-2)
  k_out = torch.stack(
      [k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], -1
  ).flatten(-2)

  return q_out.type_as(q), k_out.type_as(k)


class _VanillaTransformerEncoderLayer(nn.Module):
  """A Transformer Encoder Layer with RoPE support."""

  def __init__(
      self,
      d_model: int,
      dim_feedforward: int,
      nhead: int,
      dropout: float,
  ):
    super().__init__()
    self.d_model = d_model
    self.nhead = nhead
    self.head_dim = d_model // nhead
    if self.head_dim * nhead != self.d_model:
      raise ValueError(
          f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
      )
    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.attn_dropout_p = dropout  # For F.scaled_dot_product_attention

    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.activation = nn.ReLU()
    self.ff_dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

  def _sa_block(
      self,
      x: torch.Tensor,
      rotary_pos_emb: _RotaryPositionalEmbedding,
      key_padding_mask: torch.Tensor | None,
  ) -> torch.Tensor:
    """Self-attention block with RoPE, manual projection, and correct masking."""
    B, L, _ = x.shape  # pylint: disable=invalid-name

    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # Reshape for multi-head attention: (B, L, D) -> (B, H, L, D_h)
    q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
    k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
    v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    cos, sin = rotary_pos_emb(q)
    q, k = _apply_rotary_pos_emb(q, k, cos, sin)

    final_attn_mask = (
        key_padding_mask.view(B, 1, 1, L)
        if key_padding_mask is not None
        else None
    )
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=final_attn_mask,
        dropout_p=self.attn_dropout_p if self.training else 0.0,
    )

    # Reshape and apply output projection
    attn_output = (
        attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
    )
    return self.out_proj(attn_output)

  def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear2(self.ff_dropout(self.activation(self.linear1(x))))

  def forward(
      self,
      src: torch.Tensor,
      rotary_pos_emb: _RotaryPositionalEmbedding,
      src_key_padding_mask: torch.Tensor | None,
  ) -> torch.Tensor:
    """Applies the encoder layer with the Pre-Norm structure."""
    x = src
    x = x + self.dropout1(
        self._sa_block(self.norm1(x), rotary_pos_emb, src_key_padding_mask)
    )
    return x + self.dropout2(self._ff_block(self.norm2(x)))


class VanillaEncoder(BaseEncoder):
  """A stack of vanilla Transformer Encoder Layers."""

  def __init__(
      self,
      vocab_size: int,
      d_model: int,
      num_layers: int,
      max_len: int,
      expand: int = 4,
      nhead: int = 8,
      dropout: float = 0.0,
  ):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.d_model = d_model
    self.rotary_pos_emb = _RotaryPositionalEmbedding(
        d_model // nhead, max_len=max_len
    )

    layer = _VanillaTransformerEncoderLayer(
        d_model, expand * d_model, nhead, dropout
    )
    self.layers = nn.ModuleList(
        [copy.deepcopy(layer) for _ in range(num_layers)]
    )
    self.num_layers = num_layers
    self.norm = nn.LayerNorm(d_model)

  def forward(
      self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor | None
  ) -> torch.Tensor:
    """Forward pass conforms to the BaseEncoder interface."""
    output = self.embedding(src_ids)
    for mod in self.layers:
      output = mod(
          output, self.rotary_pos_emb, src_key_padding_mask=src_key_padding_mask
      )
    return self.norm(output)

  @property
  def hidden_dim(self) -> int:
    return self.d_model


class MambaEncoder(BaseEncoder):
  """Encoder built from a stack of Mamba blocks."""

  def __init__(
      self, vocab_size: int, d_model: int, num_layers: int, **mamba_kwargs
  ):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.d_model = d_model
    self.layers = nn.ModuleList([
        nn.Sequential(
            nn.LayerNorm(d_model),
            mamba_ssm.Mamba2(d_model=d_model, **mamba_kwargs),  # pytype: disable=attribute-error
        )
        for _ in range(num_layers)
    ])
    self.norm = nn.LayerNorm(d_model)

  def forward(
      self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor | None
  ) -> torch.Tensor:
    output = self.embedding(src_ids)
    for layer in self.layers:
      output = layer(output) + output  # Residual connection

    # Manually apply padding mask
    if src_key_padding_mask is not None:
      output = output.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
    return self.norm(output)

  @property
  def hidden_dim(self) -> int:
    return self.d_model


def _sample_orthogonal_q(
    cols: int, device: torch.device | None = None
) -> torch.Tensor:
  unstructured_block = torch.randn((cols, cols), device=device)
  q, _ = torch.linalg.qr(unstructured_block.cpu(), mode="reduced")
  return q.to(device).t()


def _gaussian_orthogonal_random_matrix(
    nb_rows: int,
    nb_columns: int,
    sqrt_scaling: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
  """Creates a gaussian orthogonal random matrix."""
  nb_full_blocks = int(nb_rows / nb_columns)
  block_list = []
  for _ in range(nb_full_blocks):
    q = _sample_orthogonal_q(nb_columns, device=device)
    block_list.append(q)
  remaining_rows = nb_rows - nb_full_blocks * nb_columns
  if remaining_rows > 0:
    q = _sample_orthogonal_q(nb_columns, device=device)
    block_list.append(q[:remaining_rows])
  final_matrix = torch.cat(block_list)

  if sqrt_scaling:
    multiplier = math.sqrt(float(nb_columns))
    multiplier = multiplier * torch.ones((nb_rows,), device=device)
  else:
    multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)

  return torch.diag(multiplier) @ final_matrix


def _generic_kernel_transformation(
    data: torch.Tensor,
    is_query: bool,
    projection_matrix: torch.Tensor | None,
    activation_fn: Callable[[torch.Tensor], torch.Tensor],
    numerical_stabilizer: float = 1e-3,
) -> torch.Tensor:
  """Computes features based on an activation function (e.g., ReLU)."""
  del is_query
  if projection_matrix is None:
    return activation_fn(data) + numerical_stabilizer

  ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
  data_dash = torch.einsum("...hd,md->...hm", data, projection_matrix)
  return activation_fn(ratio * data_dash) + numerical_stabilizer


def _exp_softmax_kernel_transformation(
    data: torch.Tensor,
    is_query: bool,
    projection_matrix: torch.Tensor,
    numerical_stabilizer: float = 1e-6,
) -> torch.Tensor:
  """Computes random features for the softmax kernel using FAVOR+ mechanism."""
  data_normalizer = 1.0 / (data.shape[-1] ** 0.25)
  ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
  data_dash = torch.einsum(
      "...hd,md->...hm", data_normalizer * data, projection_matrix
  )

  diag_data = torch.sum(data**2, dim=-1)
  diag_data = (diag_data / 2.0) * (data_normalizer**2)
  diag_data = diag_data.unsqueeze(dim=-1)

  if is_query:
    # For queries, normalization is applied per-token
    stab = torch.amax(data_dash, dim=-1, keepdim=True).detach()
  else:
    # For keys, normalization is applied across all tokens
    stab = torch.amax(data_dash, dim=(-2, -1), keepdim=True).detach()

  return ratio * (
      torch.exp(data_dash - stab - diag_data) + numerical_stabilizer
  )


def _favor_numerator(
    qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor
) -> torch.Tensor:
  """Computes not-normalized FAVOR+ attention numerator: Q'(K'T V)."""
  kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
  return torch.einsum("blhm,bhmd->blhd", qs, kvs)


def _favor_denominator(qs: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
  """Computes FAVOR+ attention denominator: Q'(K'T 1)."""
  ks_sum = torch.sum(ks, dim=1)  # Sum over the length dimension
  return torch.einsum("blhm,bhm->blh", qs, ks_sum)


def _favor_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_fn: Callable[
        [torch.Tensor, bool, torch.Tensor | None], torch.Tensor
    ],
    projection_matrix: torch.Tensor,
    inputs_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Computes bidirectional (noncausal) normalized FAVOR+ attention."""
  query_prime = kernel_fn(query, True, projection_matrix)
  key_prime = kernel_fn(key, False, projection_matrix)

  if inputs_mask is not None:  # (B, L) -> (B, L, H, M) for key_prime
    key_prime = key_prime.masked_fill(~inputs_mask[:, :, None, None], 0.0)

  # Compute numerator and denominator
  numerator = _favor_numerator(query_prime, key_prime, value)
  denominator = _favor_denominator(query_prime, key_prime)
  denominator = denominator.unsqueeze(-1) + 1e-6
  return numerator / denominator


class _PerformerEncoderLayer(nn.Module):
  """A Transformer Encoder Layer using the FAVOR+ attention mechanism."""

  def __init__(
      self,
      d_model: int,
      nhead: int,
      dim_feedforward: int,
      kernel_name: Literal["softmax", "relu"],
      num_features: int | None,
      dropout: float,
  ):
    super().__init__()
    self.d_model = d_model
    self.nhead = nhead
    self.head_dim = d_model // nhead
    if self.head_dim * nhead != self.d_model:
      raise ValueError("d_model must be divisible by nhead")

    if num_features is None:
      num_features = 2 * int(self.head_dim * math.log(self.head_dim))
    self.num_features = num_features

    if kernel_name == "softmax":
      self.kernel_fn = _exp_softmax_kernel_transformation
    elif kernel_name == "relu":
      self.kernel_fn = functools.partial(
          _generic_kernel_transformation, activation_fn=F.relu
      )
    else:
      raise ValueError(f"Unknown kernel transformation: {kernel_name}")

    # Create and register the random projection matrix
    projection_matrix = _gaussian_orthogonal_random_matrix(
        self.num_features,
        self.head_dim,
        sqrt_scaling=(kernel_name == "softmax"),
    )
    self.register_buffer("projection_matrix", projection_matrix)

    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.activation = nn.ReLU()
    self.ff_dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

  @torch.no_grad()
  def redraw_projection_matrix(
      self, device: torch.device | None = None
  ) -> None:
    """Redraws the random projection matrix for FAVOR+ attention."""
    device = device if device is not None else self.projection_matrix.device
    new_projection_matrix = _gaussian_orthogonal_random_matrix(
        self.num_features, self.head_dim, device=device
    )
    self.projection_matrix.copy_(new_projection_matrix)
    del new_projection_matrix

  def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear2(self.ff_dropout(self.activation(self.linear1(x))))

  def forward(
      self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None
  ) -> torch.Tensor:
    x = src
    x_norm = self.norm1(x)

    q = self.q_proj(x_norm)
    k = self.k_proj(x_norm)
    v = self.v_proj(x_norm)

    # Reshape for multi-head: (B, L, D) -> (B, L, H, D_h)
    q = q.view(-1, x.shape[1], self.nhead, self.head_dim)
    k = k.view(-1, x.shape[1], self.nhead, self.head_dim)
    v = v.view(-1, x.shape[1], self.nhead, self.head_dim)

    # Create mask for favor_attention (True for real tokens)
    mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
    attn_output = _favor_attention(
        q,
        k,
        v,
        kernel_fn=self.kernel_fn,
        projection_matrix=self.projection_matrix,
        inputs_mask=mask,
    )
    attn_output = attn_output.contiguous().view(-1, x.shape[1], self.d_model)
    attn_output = self.out_proj(attn_output)

    x = x + self.dropout1(attn_output)
    return x + self.dropout2(self._ff_block(self.norm2(x)))


class PerformerEncoder(BaseEncoder):
  """Encoder built from our custom PerformerEncoderLayer."""

  def __init__(
      self,
      vocab_size: int,
      d_model: int,
      num_layers: int,
      kernel_name: Literal["softmax", "relu"] = "relu",
      num_features: int | None = None,
      redraw_interval: int | None = None,
      nhead: int = 8,
      dropout: float = 0.0,
  ):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.d_model = d_model
    layer = _PerformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward=4 * d_model,
        kernel_name=kernel_name,
        num_features=num_features,
        dropout=dropout,
    )
    self.layers = nn.ModuleList(
        [copy.deepcopy(layer) for _ in range(num_layers)]
    )
    self.redraw_interval = redraw_interval
    self.register_buffer("training_calls", torch.tensor(0))
    self.norm = nn.LayerNorm(d_model)

  def forward(
      self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor | None
  ) -> torch.Tensor:
    src = self.embedding(src_ids)

    if self.training and self.redraw_interval is not None:
      self.training_calls += 1
      if self.training_calls.item() % self.redraw_interval == 0:
        for layer in self.layers:
          layer.redraw_projection_matrix(src.device)

    output = src
    for layer in self.layers:
      output = layer(output, src_key_padding_mask)
    return self.norm(output)

  @property
  def hidden_dim(self) -> int:
    return self.d_model


class T5GemmaEncoder(BaseEncoder):
  """Wrapper around a Hugging Face T5Gemma encoder."""

  def __init__(
      self,
      vocab_size: int,
      model_name: str,
      random_init: bool = False,
      freeze_weights: bool = False,
      attn_implementation: str = "sdpa",  # Flash causes issues ATM.
      dropout: float = 0.0,
  ):
    super().__init__()

    if not random_init:  # Pretrained model.
      config = transformers.AutoConfig.from_pretrained(model_name)
      config.dropout_rate = dropout
      config.attention_dropout = dropout
      model = transformers.T5GemmaForConditionalGeneration.from_pretrained(
          model_name,
          config=config,
          attn_implementation=attn_implementation,
      )
      if vocab_size != model.config.vocab_size:
        logging.info(
            "NOTE: Pretrained T5Gemma vocab resized from %d to %d.",
            model.config.vocab_size,
            vocab_size,
        )
        model.resize_token_embeddings(vocab_size)
    else:  # Random initialization.
      config = transformers.AutoConfig.from_pretrained(model_name)
      config.vocab_size = vocab_size
      config.dropout_rate = dropout
      config.attention_dropout = dropout
      model = transformers.T5GemmaForConditionalGeneration(config)

    self.encoder = model.get_encoder()

    if freeze_weights:
      for param in self.encoder.parameters():
        param.requires_grad = False

  def forward(
      self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor | None
  ) -> torch.Tensor:
    attention_mask = (
        ~src_key_padding_mask if src_key_padding_mask is not None else None
    )
    enc_out = self.encoder(
        input_ids=src_ids, attention_mask=attention_mask, return_dict=True
    )
    return enc_out.last_hidden_state

  @property
  def hidden_dim(self) -> int:
    return self.encoder.config.hidden_size


@enum.unique
class EncoderType(enum.Enum):
  """Defines the supported encoder architectures."""

  VANILLA = enum.auto()
  MAMBA = enum.auto()
  PERFORMER = enum.auto()
  T5GEMMA = enum.auto()

  def make(
      self,
      vocab_size: int,
      d_model: int,
      num_layers: int,
      max_encoder_len: int,
      **encoder_kwargs,
  ) -> BaseEncoder:
    """Factory function to make the specified encoder."""

    if self is EncoderType.VANILLA:
      return VanillaEncoder(
          vocab_size=vocab_size,
          d_model=d_model,
          num_layers=num_layers,
          max_len=max_encoder_len,
          **encoder_kwargs,
      )
    elif self is EncoderType.MAMBA:
      return MambaEncoder(
          vocab_size=vocab_size,
          d_model=d_model,
          num_layers=num_layers,
          **encoder_kwargs,
      )
    elif self is EncoderType.PERFORMER:
      return PerformerEncoder(
          vocab_size=vocab_size,
          d_model=d_model,
          num_layers=num_layers,
          **encoder_kwargs,
      )
    elif self is EncoderType.T5GEMMA:
      return T5GemmaEncoder(vocab_size=vocab_size, **encoder_kwargs)
    else:
      raise NotImplementedError(f"Encoder type {self} is not implemented.")
