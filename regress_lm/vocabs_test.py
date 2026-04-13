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

"""Tests for vocab classes."""

import pathlib
from regress_lm import vocabs
from absl.testing import absltest


class BasicEnglishVocabTest(absltest.TestCase):

  def test_basic_runs(self):
    vocab = vocabs.BasicEnglishVocab(["hello", "world"])
    self.assertEqual(vocab.pad_id, 0)
    self.assertLen(vocab, 4)
    self.assertEqual(vocab.to_token_ids("hello world foo"), [2, 3, 1])


class StructuredTextVocabTest(absltest.TestCase):

  def test_basic_runs(self):
    vocab = vocabs.StructuredTextVocab(
        tokens=["{", "}", "[", "]", ",", ":", "abc", "xyz"],
        split_regex=r"(:)",
    )
    self.assertEqual(vocab.pad_id, 0)
    self.assertLen(vocab, 10)
    # TODO: pre_tokenizer is being completely ignored here.
    self.assertEqual(vocab.to_token_ids("{abc:xyz,xyz:abc}"), [1])


class SentencePieceVocabTest(absltest.TestCase):

  def test_basic_runs(self):
    vocab = vocabs.SentencePieceVocab.from_t5()
    self.assertEqual(vocab.pad_id, 0)
    self.assertLen(vocab, 32100)
    self.assertEqual(vocab.to_token_ids("hello world"), [21820, 296])


class XValVocabWrapperTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.base_vocab = vocabs.SentencePieceVocab.from_t5()
    self.xval = vocabs.XValVocabWrapper(self.base_vocab)

  def test_mixed_text_and_numbers(self):
    """Numbers in text are replaced; surrounding text is tokenized normally."""
    feat = self.xval.to_features("temp=42.0 pressure=101.3")
    for is_f, fval in zip(feat["is_float"], feat["float_value"]):
      if is_f:
        self.assertIn(fval, [42.0, 101.3])
      else:
        self.assertEqual(fval, 0.0)

  def test_negative_and_scientific(self):
    """Negative numbers and scientific notation should be handled."""
    feat = self.xval.to_features("-5.0 1e3")
    float_vals = [v for f, v in zip(feat["is_float"], feat["float_value"]) if f]
    self.assertIn(-5.0, float_vals)
    self.assertIn(1000.0, float_vals)

  def test_multiple_numbers_one_token_each(self):
    """Every number is always exactly one token, regardless of value."""
    feat = self.xval.to_features("1e-8 42.0 1e8")
    num_positions = sum(feat["is_float"])
    self.assertEqual(num_positions, 3)
    # All number positions should share the same <num> token ID.
    num_ids = [i for i, f in zip(feat["ids"], feat["is_float"]) if f]
    self.assertLen(set(num_ids), 1)

  def test_vocab_size(self):
    """Vocab size is always base + 1 (single <num> token)."""
    self.assertLen(self.xval, len(self.base_vocab) + 1)

  def test_feature_alignment(self):
    """ids, is_float, float_value must be aligned for mixed text+numbers."""
    text = "temp 98.6 pressure 101.3 at -0.5"
    feat = self.xval.to_features(text)
    num_id = len(self.base_vocab)  # The single <num> token ID.

    # 1. All three lists must be the same length.
    self.assertLen(feat["is_float"], len(feat["ids"]))
    self.assertLen(feat["float_value"], len(feat["ids"]))

    # 2. Walk through and verify each position.
    numbers_found = []
    text_segments = []
    for i, (tid, is_f, fval) in enumerate(
        zip(feat["ids"], feat["is_float"], feat["float_value"])
    ):
      if is_f:
        # Number position: ID must be <num>, value must be the number.
        self.assertEqual(tid, num_id, f"Position {i}: expected <num> ID.")
        numbers_found.append(fval)
      else:
        # Text position: ID must be a base vocab ID, value must be 0.
        self.assertLess(tid, num_id, f"Position {i}: expected base vocab ID.")
        self.assertEqual(fval, 0.0, f"Position {i}: text should have 0.0.")
        text_segments.append(tid)

    # 3. Verify the correct numbers were extracted in order.
    self.assertLen(numbers_found, 3)
    self.assertAlmostEqual(numbers_found[0], 98.6)
    self.assertAlmostEqual(numbers_found[1], 101.3)
    self.assertAlmostEqual(numbers_found[2], -0.5)

    # 4. Text tokens should match base vocab tokenization of the text parts.
    expected_text_ids = (
        self.base_vocab.to_token_ids("temp ")
        + self.base_vocab.to_token_ids(" pressure ")
        + self.base_vocab.to_token_ids(" at ")
    )
    self.assertEqual(text_segments, expected_text_ids)

  def test_token_count_reduction(self):
    """xVal should use far fewer tokens for number-heavy text.

    SentencePiece splits each digit into a separate token, so a number
    like "12345.6789" becomes ~10+ tokens. xVal replaces it with 1 token.
    """
    numeric_text = "1.23 4.56 7.89 0.001 123456.789 -3.14e10"
    base_ids = self.base_vocab.to_token_ids(numeric_text)
    xval_feat = self.xval.to_features(numeric_text)
    reduction = len(base_ids) / len(xval_feat["ids"])
    # Should be a significant reduction (at least 2x).
    self.assertGreater(
        reduction,
        2.0,
        f"Expected >2x token reduction, got {reduction:.1f}x "
        f"({len(base_ids)} -> {len(xval_feat['ids'])} tokens)",
    )


if __name__ == "__main__":
  absltest.main()
