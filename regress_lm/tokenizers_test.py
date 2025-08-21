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

import math
from regress_lm import tokenizers
from absl.testing import absltest
from absl.testing import parameterized


class P10TokenizerTest(parameterized.TestCase):

  @parameterized.parameters(
      (123.4, '<+><1><2><3><4><E-1>', 123.4),
      (12345, '<+><1><2><3><4><E1>', 12340),
      (0.1234, '<+><1><2><3><4><E-4>', 0.1234),
      (-123.4, '<-><1><2><3><4><E-1>', -123.4),
      (-12345, '<-><1><2><3><4><E1>', -12340),
      (-0.1234, '<-><1><2><3><4><E-4>', -0.1234),
      (0.0, '<+><0><0><0><0><E0>', 0.0),
      (-0.0, '<+><0><0><0><0><E0>', 0.0),  # in python, 0.0 == -0.0
      (-0.4e-13, '<-><0><0><0><0><E0>', 0.0),  # note leading negative zero
  )
  def test_tokenize(self, f: float, tokens: str, f_prime: float):
    vocab = tokenizers.P10Tokenizer()
    out_tokens = vocab.to_tokens(f)
    self.assertEqual(''.join(out_tokens), tokens)
    self.assertAlmostEqual(vocab.from_tokens(out_tokens), f_prime)

  @parameterized.parameters((3, 10, 1.0e-8, 9.99e12), (1, 5, 1.0e-5, 9.0e5))
  def test_representation_range(
      self,
      num_digits: int,
      exponent_range: int,
      min_val: float,
      max_val: float,
  ):
    vocab = tokenizers.P10Tokenizer(
        num_digits=num_digits,
        exponent_range=exponent_range,
    )
    self.assertEqual(vocab._max_abs_val, max_val)
    self.assertEqual(vocab._min_abs_val, min_val)

  @parameterized.parameters(
      (1.0e13, 3, 10, '<+><9><9><9><E10>'),
      (2.0e13, 3, 10, '<+><9><9><9><E10>'),
      (-1.0e13, 3, 10, '<-><9><9><9><E10>'),
      (-2.0e13, 3, 10, '<-><9><9><9><E10>'),
      (9.9e12, 3, 10, '<+><9><9><0><E10>'),
      (-9.9e12, 3, 10, '<-><9><9><0><E10>'),
      (5.0e5, 3, 10, '<+><5><0><0><E3>'),
      (1.1e-8, 3, 10, '<+><1><1><0><E-10>'),
      (0.9e-8, 3, 10, '<+><1><0><0><E-10>'),
      (0.5e-8, 3, 10, '<+><0><0><0><E0>'),
      (0.51e-8, 3, 10, '<+><1><0><0><E-10>'),
      (0.4e-8, 3, 10, '<+><0><0><0><E0>'),
      # rounding up below creats a negative sign for 0
      (-0.4e-8, 3, 10, '<-><0><0><0><E0>'),
      (-0.5e-8, 3, 10, '<-><0><0><0><E0>'),
      (-0.51e-8, 3, 10, '<-><1><0><0><E-10>'),
      (-0.8e-8, 3, 10, '<-><1><0><0><E-10>'),
      (-1.1e-8, 3, 10, '<-><1><1><0><E-10>'),
  )
  def test_tokenization_limit(
      self,
      f: float,
      num_digits: int,
      exponent_range: int,
      serialized: str,
  ):
    vocab = tokenizers.P10Tokenizer(
        num_digits=num_digits,
        exponent_range=exponent_range,
    )
    self.assertEqual(''.join(vocab.to_tokens(f)), serialized)

  def test_all_tokens_used(self):
    vocab = tokenizers.P10Tokenizer(exponent_range=2)
    out = vocab.all_tokens()

    signs = ['<+>', '<->']
    digits = [f'<{i}>' for i in range(10)]
    exponents = ['<E-2>', '<E-1>', '<E0>', '<E1>', '<E2>']
    self.assertEqual(list(out), signs + digits + exponents)


class IEEEFloatTokenizerTest(parameterized.TestCase):

  @parameterized.parameters(
      (123.4, '<+><+><2><1><2><3><4>', 123.4),
      (12345, '<+><+><4><1><2><3><4>', 12340),
      (0.1234, '<+><-><1><1><2><3><4>', 0.1234),
      (-123.4, '<-><+><2><1><2><3><4>', -123.4),
      (-12345, '<-><+><4><1><2><3><4>', -12340),
      (-0.1234, '<-><-><1><1><2><3><4>', -0.1234),
      (1.234e-9, '<+><-><9><1><2><3><4>', 1.234e-9),
      (-1.234e-9, '<-><-><9><1><2><3><4>', 1.234e-9),
      (1.2e-9, '<+><-><9><1><2><0><0>', 1.2e-9),
      (-1.2e-9, '<-><-><9><1><2><0><0>', -1.2e-9),
      (1.2e9, '<+><+><9><1><2><0><0>', 1.2e9),
      (-1.2e9, '<-><+><9><1><2><0><0>', -1.2e9),
      (1.2345e9, '<+><+><9><1><2><3><4>', 1.234e9),
      (1.234e-10, '<+><-><0><0><0><0><0>', 0.0),  # Underflow
      (-1.234e-10, '<-><-><0><0><0><0><0>', 0.0),  # Underflow
      (0.0, '<+><+><0><0><0><0><0>', 0.0),
      (-0.0, '<+><+><0><0><0><0><0>', 0.0),  # in python, 0.0 == -0.0
  )
  def test_tokenize(self, f: float, tokens: str, f_prime: float):
    vocab = tokenizers.IEEEFloatTokenizer()
    out_tokens = vocab.to_tokens(f)
    self.assertEqual(''.join(out_tokens), tokens)
    self.assertAlmostEqual(vocab.from_tokens(out_tokens), f_prime)

  @parameterized.parameters((3,), (10,), (18,))
  def test_all_tokens_used(self, base: int):
    vocab = tokenizers.IEEEFloatTokenizer(base=base)
    out = vocab.all_tokens()

    signs = ['<+>', '<->']
    digits = [f'<{i}>' for i in range(base)]
    self.assertEqual(list(out), signs + digits)


class AddSpecialValuesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.base_tokenizer = tokenizers.P10Tokenizer(
        num_digits=1, exponent_range=2
    )
    self.tokenizer = tokenizers.AddSpecialValues(self.base_tokenizer)

  @parameterized.parameters(
      ('INVALID', ['<INVALID>', '<INVALID>', '<INVALID>']),
      (float('nan'), ['<NAN>', '<NAN>', '<NAN>']),
      (float('inf'), ['<INF>', '<INF>', '<INF>']),
      (float('-inf'), ['<NINF>', '<NINF>', '<NINF>']),
      (100.0, ['<+>', '<1>', '<E2>']),
  )
  def test_tokenize(self, f: float, expected_tokens: list[str]):
    out_tokens = self.tokenizer.to_tokens(f)
    self.assertEqual(out_tokens, expected_tokens)
    f_prime = self.tokenizer.from_tokens(out_tokens)
    if f_prime == 'INVALID':
      self.assertEqual(f_prime, 'INVALID')
    elif math.isnan(f):
      self.assertTrue(math.isnan(f_prime))
    else:
      self.assertAlmostEqual(f_prime, f)

  def test_all_tokens_used(self):
    self.assertEqual(
        list(self.tokenizer.all_tokens()),
        ['<+>', '<->']
        + [f'<{i}>' for i in range(10)]
        + ['<E-2>', '<E-1>', '<E0>', '<E1>', '<E2>']
        + ['<INVALID>', '<NAN>', '<INF>', '<NINF>'],
    )

  def test_possible_next_tokens(self):
    self.assertEqual(
        self.tokenizer.possible_next_tokens([]),
        ['<INVALID>', '<NAN>', '<INF>', '<NINF>'] + ['<+>', '<->'],
    )

    self.assertEqual(
        self.tokenizer.possible_next_tokens(['<+>']),
        [f'<{i}>' for i in range(10)],
    )

    self.assertEqual(
        self.tokenizer.possible_next_tokens(['<INVALID>']), ['<INVALID>']
    )


if __name__ == '__main__':
  absltest.main()
