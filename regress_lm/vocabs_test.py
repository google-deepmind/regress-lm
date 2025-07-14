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


if __name__ == "__main__":
  absltest.main()
