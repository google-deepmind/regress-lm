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
import math
import re
from typing import Generic, TypeVar, Union

import numpy as np
import ordered_set

OrderedSet = ordered_set.OrderedSet
DELIMITERS = ('<', '>')
INFEASIBLE = 'INFEASIBLE'

SpecialFPTokenValue = Union[str, float]
SPECIAL_TOKENS: dict[str, SpecialFPTokenValue] = {
    'INFEASIBLE': INFEASIBLE,
    'NAN': float('nan'),
    'INF': float('inf'),
    'NINF': float('-inf'),
}


def _get_special_token_str(v: SpecialFPTokenValue) -> str:
  if v == INFEASIBLE:
    return INFEASIBLE
  if math.isnan(v):
    return 'NAN'
  if math.isinf(v):
    return 'INF' if v > 0 else 'NINF'
  raise ValueError(f'Not a special float: {v}')


def _is_special_fp_token(v: SpecialFPTokenValue) -> bool:
  if isinstance(v, float) and math.isnan(v):
    return True
  return v in SPECIAL_TOKENS.values()


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


_SPECIAL_TOKEN_SET = OrderedSet([_to_token(s) for s in SPECIAL_TOKENS])
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


class AddSpecialTokens(DecoderTokenizer[SpecialFPTokenValue]):
  """Wraps a DecoderTokenizer to add special float tokens."""

  def __init__(self, tokenizer: DecoderTokenizer[SpecialFPTokenValue]):
    self._tokenizer = tokenizer

  @property
  def num_tokens_per_obj(self) -> int:
    return self._tokenizer.num_tokens_per_obj

  def all_tokens(self) -> OrderedSet[str]:
    return self._tokenizer.all_tokens().union(_SPECIAL_TOKEN_SET)

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    index = len(prev_tokens) % self.num_tokens_per_obj
    if index == 0:
      return _SPECIAL_TOKEN_SET.union(
          self._tokenizer.possible_next_tokens(prev_tokens)
      )
    zeroth_token = prev_tokens[0]
    return (
        OrderedSet([zeroth_token])
        if _from_token(zeroth_token) in SPECIAL_TOKENS
        else self._tokenizer.possible_next_tokens(prev_tokens)
    )

  def to_tokens(self, obj: SpecialFPTokenValue, /) -> list[str]:
    if _is_special_fp_token(obj):
      return [_to_token(_get_special_token_str(obj))] * self.num_tokens_per_obj
    return self._tokenizer.to_tokens(obj)

  def from_tokens(self, tokens: list[str], /) -> SpecialFPTokenValue:
    primitive = _from_token(tokens[0])
    if primitive in SPECIAL_TOKENS:
      return SPECIAL_TOKENS[primitive]
    return self._tokenizer.from_tokens(tokens)


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
    if num_digits <= 0:
      raise ValueError('num_digits must be positive.')
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
    return 2 + self.num_digits  # sign + mantissa

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    index = len(prev_tokens) % self.num_tokens_per_obj
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

  @property
  def _max_abs_val(self) -> float:
    """Largest representable positive number."""
    return float(self.num_digits * '9') * (10.0**self.exponent_range)

  @property
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
    return OrderedSet(
        [_to_token(s) for s in ['+', '-'] + list(range(self.base))]
    )

  @property
  def num_tokens_per_obj(self) -> int:
    return 2 + self.num_exponent_digits + self.num_mantissa_digits

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    index = len(prev_tokens) % self.num_tokens_per_obj
    if index < 0 or index >= self.num_tokens_per_obj:
      raise ValueError(f'Index {index} out of bounds.')

    if index in [0, 1]:  # beginning
      tokens = [_to_token(s) for s in ['+', '-']]
    else:  # middle (digit)
      tokens = [_to_token(s) for s in range(self.base)]
    return OrderedSet(tokens)

  def to_tokens(self, f: float, /) -> list[str]:
    sign = '+' if f >= 0 else '-'
    abs_f = abs(f)
    exponent = math.floor(np.log(abs_f) / np.log(self.base)) if abs_f > 0 else 0
    exponent_sign = '+' if exponent >= 0 else '-'
    abs_exponent = abs(exponent)

    e = np.base_repr(abs_exponent, base=self.base)
    if len(e) > self.num_exponent_digits and exponent_sign == '+':
      raise ValueError(f'Overflow: Exponent {abs_exponent} too large.')
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
