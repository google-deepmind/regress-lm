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

"""Custom tokenizers for RegressLM.

For full generality, note that the tokens are strings, as we assume the
vocabulary will assign integer IDs.
"""

import abc
import functools
import math
import re
from typing import Generic, TypeVar
import numpy as np
import ordered_set

OrderedSet = ordered_set.OrderedSet
DELIMITERS = ('<', '>')


def _to_token(s: str | int) -> str:
  left_d, right_d = DELIMITERS
  return f'{left_d}{s}{right_d}'


def _from_token(token: str) -> str | int:
  left_d, right_d = DELIMITERS
  pattern = f'{left_d}(.*?){right_d}'
  m = re.fullmatch(pattern, token)
  if not m:
    raise ValueError(f'Could not deserialize `{token}`.')
  return m.group(1)


ObjectT = TypeVar('ObjectT')


class DecoderTokenizer(abc.ABC, Generic[ObjectT]):
  """Abstract class for decoder tokenizers."""

  @property
  @abc.abstractmethod
  def num_tokens_per_obj(self) -> int:
    """Number of tokens used to represent each float."""

  @abc.abstractmethod
  def all_tokens(self) -> OrderedSet[str]:
    """Returns ordered set of all tokens used."""

  @abc.abstractmethod
  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    """Returns ordered set of tokens possible for the next step."""

  @abc.abstractmethod
  def to_tokens(self, obj: ObjectT, /) -> list[str]:
    """Converts an object to a string of tokens."""

  @abc.abstractmethod
  def from_tokens(self, tokens: list[str], /) -> ObjectT:
    """Converts a string of tokens to an object."""


class P10Tokenizer(DecoderTokenizer[float]):
  """Uses P10 tokenization from https://arxiv.org/abs/2112.01898.

  A float f can be represented as:

  `s * m * 10^e`

  where:
    s: Positive/Negative sign (+, -)
    m: Mantissa representing leading digits.
    e: Exponent.

  Attributes:
    num_digits: Number of digits in `m`. Each digit (even the leading) is
      between <0> and <9>.
    exponent_range: Controls number of exponent tokens, e.g. if 10, the exponent
      token range will be [<E-10>, <E10>], affecting the range of representable
      floats.
  """

  def __init__(self, num_digits: int = 4, exponent_range: int = 10):
    self.num_digits = num_digits
    self.exponent_range = exponent_range

  def all_tokens(self) -> OrderedSet[str]:
    tokens = [_to_token(s) for s in ['+', '-'] + list(range(10))]
    exps = [
        f'E{i}' for i in range(-self.exponent_range, self.exponent_range + 1)
    ]
    tokens.extend([_to_token(s) for s in exps])
    return OrderedSet(tokens)

  @property
  def num_tokens_per_obj(self) -> int:
    return 2 + self.num_digits

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    index = len(prev_tokens)
    if index < 0 or index >= self.num_tokens_per_obj:
      raise ValueError(f'Index {index} out of bounds.')

    if index == 0:  # beginning
      tokens = [_to_token(s) for s in ['+', '-']]
    elif index == self.num_tokens_per_obj - 1:  # end
      exps = [
          f'E{i}' for i in range(-self.exponent_range, self.exponent_range + 1)
      ]
      tokens = [_to_token(s) for s in exps]
    else:  # middle (digit)
      tokens = [_to_token(s) for s in range(0, 10)]
    return OrderedSet(tokens)

  @functools.cached_property
  def _max_abs_val(self) -> float:
    """Largest representable positive number."""
    return float(self.num_digits * '9') * (10.0**self.exponent_range)

  @functools.cached_property
  def _min_abs_val(self) -> float:
    """Smallest representable positive number."""
    min_mantissa = float('1' + (self.num_digits - 1) * '0')
    return min_mantissa * (10 ** (-self.exponent_range))

  def _round_float(self, f: float) -> float:
    """Rounds float to the closest in-range value."""
    abs_f = abs(f)
    abs_f = min(abs_f, self._max_abs_val)
    if abs_f < self._min_abs_val:
      # Decides whether to move to 0.0 or `min_abs_val`.
      zero_or_min = round(abs_f / self._min_abs_val)
      abs_f = self._min_abs_val * zero_or_min
    return abs_f if f >= 0 else -abs_f

  def to_tokens(self, f: float, /) -> list[str]:
    f = self._round_float(f)
    s = np.format_float_scientific(
        f,
        precision=self.num_digits - 1,
        min_digits=self.num_digits - 1,
        sign=True,
    )
    # We expect numpy to produce scientific notation of the form `+2.123e+4`.
    # It will round for us and ensure leading digit isn't zero, unless the
    # number is zero.
    m = re.fullmatch('([+-])([0-9.]*)e(.*)', s)
    if not m:
      raise RuntimeError(f'Unexpected numpy notation: {s}')
    sign: str = m.group(1)
    digits = list(m.group(2).replace('.', ''))
    exp = int(m.group(3)) - len(digits) + 1 if f else 0

    out = [sign] + digits + [f'E{exp}']
    return [_to_token(s) for s in out]

  def from_tokens(self, tokens: list[str], /) -> float:
    primitives = [_from_token(t) for t in tokens]

    sign = -1 if primitives[0] == '-' else 1
    mantissa = int(''.join(map(str, primitives[1:-1])))
    exp = int(''.join(primitives[-1]).lstrip('E'))

    return float(sign * mantissa * 10**exp)


class IEEEFloatTokenizer(DecoderTokenizer[float]):
  """More official float tokenizer, minimizing the use of dedicated tokens.

  Follows IEEE-type standard.

  A float f = `s * b^e * m` can be represented as [s, e, m] from most to least
  important, where:
    s: Positive/Negative sign (+, -)
    b: Base
    e: Exponent (left-most is a sign, digits represented with base b)
    m: Mantissa (represented with base b)

  For example, 1.23456789e-222 can be represented as:

  <+><-><2><2><2><1><2><3><4>

  if b=10, num_exponent_digits=3, and num_mantissa_digits=4.
  """

  def __init__(
      self,
      base: int = 10,
      num_exponent_digits: int = 1,
      num_mantissa_digits: int = 4,
  ):
    self.base = base
    self.num_exponent_digits = num_exponent_digits
    self.num_mantissa_digits = num_mantissa_digits

  def all_tokens(self) -> OrderedSet[str]:
    tokens = ['+', '-'] + list(range(self.base))
    return OrderedSet([_to_token(s) for s in tokens])

  @property
  def num_tokens_per_obj(self) -> int:
    return 2 + self.num_exponent_digits + self.num_mantissa_digits

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    index = len(prev_tokens)
    if index < 0 or index >= self.num_tokens_per_obj:
      raise ValueError(f'Index {index} out of bounds.')

    if index in [0, 1]:  # beginning
      tokens = [_to_token(s) for s in ['+', '-']]
    else:  # middle (digit)
      tokens = [_to_token(s) for s in range(self.base)]
    return OrderedSet(tokens)

  def to_tokens(self, f: float, /) -> list[str]:
    if abs(f) > self._max_abs_val:  # Round during overflows.
      f = math.copysign(self._max_abs_val, f)

    sign = '+' if f >= 0 else '-'
    abs_f = abs(f)
    exponent = math.floor(np.log(abs_f) / np.log(self.base)) if abs_f > 0 else 0

    exponent_sign = '+' if exponent >= 0 else '-'
    abs_exponent = abs(exponent)

    e = np.base_repr(abs_exponent, base=self.base)
    if len(e) > self.num_exponent_digits and exponent_sign == '-':
      # Underflow.
      all_zeros = ['0'] * (self.num_exponent_digits + self.num_mantissa_digits)
      out = [sign, '-'] + all_zeros
      return [_to_token(s) for s in out]
    e = e.zfill(self.num_exponent_digits)

    mantissa = np.base_repr(
        abs_f * self.base ** (self.num_mantissa_digits - 1 - exponent),
        base=self.base,
    )

    if len(mantissa) > self.num_mantissa_digits:
      mantissa = mantissa[: self.num_mantissa_digits]
    if len(mantissa) < self.num_mantissa_digits:  # Right-pad with zeros.
      mantissa += '0' * (self.num_mantissa_digits - len(mantissa))

    raw_str = sign + exponent_sign + e + mantissa
    return [_to_token(s) for s in raw_str]

  def from_tokens(self, tokens: list[str], /) -> float:
    primitives = [_from_token(t) for t in tokens]

    sign = -1 if primitives[0] == '-' else 1

    exponent_sign = -1 if primitives[1] == '-' else 1

    abs_exponent_str = ''.join(
        map(str, primitives[2 : 2 + self.num_exponent_digits])
    )
    abs_exponent = int(abs_exponent_str, base=self.base)
    exponent = exponent_sign * abs_exponent

    mantissa_str = ''.join(map(str, primitives[2 + self.num_exponent_digits :]))
    mantissa_unscaled = int(mantissa_str, base=self.base)
    mantissa = mantissa_unscaled / self.base ** (self.num_mantissa_digits - 1)

    return sign * (self.base**exponent) * mantissa

  @functools.cached_property
  def _max_abs_val(self) -> float:
    """Largest representable positive number."""
    max_exponent = self.base**self.num_exponent_digits - 1
    max_mantissa_unscaled = self.base**self.num_mantissa_digits - 1
    max_mantissa = max_mantissa_unscaled / self.base ** (
        self.num_mantissa_digits - 1
    )
    return (self.base**max_exponent) * max_mantissa


class NormalizedTokenizer(DecoderTokenizer[float]):
  """Tokenizer which supports only numbers within [0,1].

  A float `f` in [0,1] is represented by a sequence of integer tokens in a
  given base. For example, with base=10 and length=3, 0.523 is represented as
  [<5>, <2>, <3>]. This is equivalent to `5 * 10**-1 + 2 * 10**-2 + 3 * 10**-3`.

  Attributes:
    base: The base for the tokenization. The vocabulary will be digits from 0 to
      base-1.
    length: The number of tokens used to represent a single float. This
      determines the precision.
  """

  def __init__(self, base: int = 10, length: int = 3):
    if not base >= 2:
      raise ValueError(f'Base must be >= 2, got {base}')
    if not length >= 1:
      raise ValueError(f'Length must be >= 1, got {length}')
    self.base = base
    self.length = length

  @property
  def num_tokens_per_obj(self) -> int:
    return self.length

  def all_tokens(self) -> OrderedSet[str]:
    return OrderedSet([_to_token(i) for i in range(self.base)])

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    if len(prev_tokens) >= self.length:
      raise ValueError(f'Index {len(prev_tokens)} out of bounds.')
    # For this tokenizer, any digit can appear at any position.
    return self.all_tokens()

  def to_tokens(self, f: float, /) -> list[str]:
    if not 0 <= f <= 1:
      raise ValueError(f'Input float must be between 0 and 1, got {f}')

    # Scale float and convert to an integer in the specified base.
    f_trunc = int(f * self.base**self.length)

    # Handle the edge case where f is exactly 1.0. The representation should
    # be the largest possible number, i.e., [base-1, base-1, ...].
    if f_trunc == self.base**self.length:
      f_trunc -= 1

    # Get the base representation, padded with zeros to the left.
    base_repr_str = np.base_repr(f_trunc, base=self.base).zfill(self.length)

    # Convert each digit character to a token, using the correct base.
    return [_to_token(int(b, self.base)) for b in base_repr_str]

  def from_tokens(self, tokens: list[str], /) -> float:
    if len(tokens) != self.length:
      raise ValueError(f'Expected {self.length} tokens, but got {len(tokens)}.')

    # Convert string tokens back to integer digits.
    token_ids = [int(str(_from_token(t))) for t in tokens]

    if not all(0 <= tid < self.base for tid in token_ids):
      raise ValueError(f'Token IDs {token_ids} out of range(0, {self.base})')

    # Calculate the float value from the integer digits.
    # The formula is: sum(d_i * base^(-(i+1))) for i in 0..length-1
    x = np.asarray(token_ids, dtype=np.float32)
    coeffs = np.power(
        self.base, -np.arange(1, self.length + 1), dtype=np.float32
    )
    return float(np.sum(x * coeffs))


class AddSpecialValues(DecoderTokenizer[str | float]):
  """Wraps a DecoderTokenizer to add special tokens."""

  SPECIAL_VALUE_MAP = {
      'INVALID': 'INVALID',
      'NAN': float('nan'),
      'INF': float('inf'),
      'NINF': float('-inf'),
  }

  def __init__(self, tokenizer: DecoderTokenizer):
    self._tokenizer = tokenizer
    self._special_tokens = OrderedSet(
        [_to_token(s) for s in self.SPECIAL_VALUE_MAP]
    )

  @property
  def num_tokens_per_obj(self) -> int:
    return self._tokenizer.num_tokens_per_obj

  def all_tokens(self) -> OrderedSet[str]:
    return self._tokenizer.all_tokens().union(self._special_tokens)

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    if not prev_tokens:  # At start, all special tokens are possible
      return self._special_tokens.union(
          self._tokenizer.possible_next_tokens(prev_tokens)
      )
    # If a sequence starts with a special token, it must be repeated
    if prev_tokens[0] in self._special_tokens:
      return OrderedSet([prev_tokens[0]])
    return self._tokenizer.possible_next_tokens(prev_tokens)

  def to_tokens(self, obj: str | float, /) -> list[str]:
    string = None
    if isinstance(obj, str) and obj == 'INVALID':
      string = 'INVALID'
    elif isinstance(obj, float):
      if math.isnan(obj):
        string = 'NAN'
      elif math.isinf(obj) and obj > 0:
        string = 'INF'
      elif math.isinf(obj) and obj < 0:
        string = 'NINF'

    if string is not None:
      return [_to_token(string)] * self.num_tokens_per_obj
    return self._tokenizer.to_tokens(obj)

  def from_tokens(self, tokens: list[str], /) -> str | float:
    token = _from_token(tokens[0])
    if token in self.SPECIAL_VALUE_MAP:
      return self.SPECIAL_VALUE_MAP[token]
    return self._tokenizer.from_tokens(tokens)


class AppendPadTokenizer(DecoderTokenizer[ObjectT]):
  """Wraps a tokenizer to append a <pad> token after each object.

  This is a simple way to create a separator for multi-objective tasks. Note
  our model will mask out any <pad> tokens when training.
  """

  def __init__(
      self, tokenizer: DecoderTokenizer[ObjectT], *, pad_token: str = '<pad>'
  ):
    self.tokenizer = tokenizer
    self.pad_token = pad_token

  @property
  def num_tokens_per_obj(self) -> int:
    return self.tokenizer.num_tokens_per_obj + 1

  def all_tokens(self) -> OrderedSet[str]:
    return self.tokenizer.all_tokens()  # '<pad>' is not included.

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    if len(prev_tokens) >= self.num_tokens_per_obj:
      raise ValueError(f'Index {len(prev_tokens)} out of bounds.')
    if len(prev_tokens) == self.num_tokens_per_obj - 1:  # Last position.
      return OrderedSet([self.pad_token])
    return self.tokenizer.possible_next_tokens(prev_tokens)  # Normal case.

  def to_tokens(self, obj: ObjectT, /) -> list[str]:
    return self.tokenizer.to_tokens(obj) + [self.pad_token]

  def from_tokens(self, tokens: list[str], /) -> ObjectT:
    if not tokens or tokens[-1] != self.pad_token:
      raise ValueError(f'Expected a "{self.pad_token}" token at the end.')
    return self.tokenizer.from_tokens(tokens[:-1])
