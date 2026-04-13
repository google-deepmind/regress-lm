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

"""Custom vocab classes for RegressLM."""

import abc
import dataclasses
import pathlib
import re
from typing import Any, Generic, Sequence, TypeVar
from regress_lm import tokenizers
import tokenizers as ht
import transformers
import sentencepiece as spp
import sentencepiece as spt

# pylint:disable=g-bare-generic


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
  """Declares the type and padding fill value for one feature key."""

  dtype: type[int | float | bool]
  padding: int | float | bool


ObjectT = TypeVar("ObjectT")


class BaseVocab(abc.ABC, Generic[ObjectT]):
  """Base class for vocabularies."""

  @abc.abstractmethod
  def to_token_ids(self, obj: ObjectT, /) -> list[int]:
    """Converts object (e.g. text) to token ids."""

  @abc.abstractmethod
  def __len__(self) -> int:
    """Returns the vocab size."""


# TODO: Consider making all encoder vocabs just `EncoderTokenizer`
# subclasses and have a single class `EncoderVocab` which just handles the
# token-id mapping.
class EncoderVocab(BaseVocab[ObjectT]):
  """Vocabulary class for encoders.

  Note we don't ever need to convert back to text.
  """

  @property
  @abc.abstractmethod
  def pad_id(self) -> int:
    """Returns the pad id."""

  @property
  def features_spec(self) -> dict[str, FeatureSpec]:
    """Declares (dtype, padding) for each feature returned by to_features()."""
    return {"ids": FeatureSpec(dtype=int, padding=self.pad_id)}

  # TODO: Detatch this class from BaseVocab and remove the need for
  # `to_token_ids`.
  def to_features(self, obj: ObjectT, /) -> dict[str, list]:
    """Returns per-token features, including token IDs.

    Must contain 'ids' (list[int]). May contain additional per-token
    features as extra keys (e.g. 'is_float', 'float_value').

    Args:
      obj: The object to convert to features.

    Returns:
      A dict with per-token features.
    """
    return {"ids": self.to_token_ids(obj)}


class DecoderVocab(BaseVocab[ObjectT]):
  """Vocabulary class for decoders.

  Supports single objective and multi-objective cases.

  For multi-objective, the output is simply the concatenation of tokens for each
  objective.
  """

  def __init__(
      self,
      tokenizer: tokenizers.DecoderTokenizer[ObjectT],
      *,
      pad_token: str = "<pad>",
  ):
    self.tokenizer = tokenizer
    self.pad_token = pad_token

    self.itos = [pad_token] + sorted(self.tokenizer.all_tokens())
    self.stoi = {token: i for i, token in enumerate(self.itos)}

  def to_token_ids(self, obj: ObjectT | Sequence[ObjectT], /) -> list[int]:
    obj = obj if isinstance(obj, Sequence) else [obj]
    all_tokens = []
    for o in obj:
      all_tokens.extend(self.tokenizer.to_tokens(o))
    return [self.stoi[t] for t in all_tokens]

  def from_token_ids(self, token_ids: Sequence[int], /) -> list[ObjectT]:
    """Converts token ids to object."""
    token_strs = [self.itos[id] for id in token_ids]

    if len(token_strs) % self.num_tokens_per_obj != 0:
      raise ValueError("Tokens not a multiple of tokens per object.")

    decoded_objs = []
    for i in range(0, len(token_strs), self.num_tokens_per_obj):
      chunk = token_strs[i : i + self.num_tokens_per_obj]
      decoded_objs.append(self.tokenizer.from_tokens(chunk))

    return decoded_objs

  def possible_next_token_ids(self, prev_tokens: Sequence[int]) -> list[int]:
    """Returns the possible token ids for the next step."""
    length = len(prev_tokens)

    if length % self.num_tokens_per_obj == 0:
      new_tokens = []
    else:
      remainder = length % self.num_tokens_per_obj
      new_tokens = [self.itos[i] for i in prev_tokens[-remainder:]]

    possible_next_tokens = self.tokenizer.possible_next_tokens(new_tokens)
    return [self.stoi[t] for t in possible_next_tokens]

  @property
  def bos_pad_id(self) -> int:
    """Returns the BOS / PAD id for the decoder."""
    return self.stoi[self.pad_token]

  @property
  def num_tokens_per_obj(self) -> int:
    """Returns the number of tokens used to represent each object."""
    return self.tokenizer.num_tokens_per_obj

  def __len__(self) -> int:
    """Returns the vocab size."""
    return len(self.stoi)


class BasicEnglishVocab(EncoderVocab[str]):
  """Basic English vocab for testing."""

  def __init__(self, words: list[str]):
    specials = ["<pad>", "<unk>"]
    # Build vocab dictionary ensuring special tokens have fixed IDs 0 and 1.
    vocab = {word: i + len(specials) for i, word in enumerate(words)}
    for i, token in enumerate(specials):
      vocab[token] = i

    # Instantiate a huggingface tokenizer with a WordLevel model
    self.tokenizer = ht.Tokenizer(
        ht.models.WordLevel(vocab=vocab, unk_token="<unk>")
    )
    self.tokenizer.normalizer = ht.normalizers.Lowercase()
    self.tokenizer.pre_tokenizer = ht.pre_tokenizers.Whitespace()

    pad_id_val = self.tokenizer.token_to_id("<pad>")
    if pad_id_val is None:
      raise ValueError("'<pad>' token not found in the vocabulary.")
    self._pad_id = pad_id_val

  def to_token_ids(self, obj: str) -> list[int]:
    return self.tokenizer.encode(obj).ids

  @property
  def pad_id(self) -> int:
    return self._pad_id

  def __len__(self) -> int:
    return self.tokenizer.get_vocab_size()


class StructuredTextVocab(EncoderVocab[str]):
  """For structured text, ideal for custom formats like JSON or DSLs.

  NOTE: Not working right now, pre_tokenizer is being completely ignored.
  """

  def __init__(self, tokens: list[str], split_regex: str = r"([\{\}\[\]:,])"):
    specials = ["<pad>", "<unk>"]

    self.vocab = {token: i + len(specials) for i, token in enumerate(tokens)}
    self.vocab.update({special: i for i, special in enumerate(specials)})

    self.tokenizer = ht.Tokenizer(
        ht.models.WordLevel(vocab=self.vocab, unk_token="<unk>")
    )
    pre_tokenizer = ht.pre_tokenizers.Split(
        pattern=split_regex, behavior="isolated"
    )
    self.tokenizer.pre_tokenizer = pre_tokenizer

  def to_token_ids(self, obj: str) -> list[int]:
    """Converts a structured string to a list of token IDs."""
    return self.tokenizer.encode(obj).ids

  @property
  def pad_id(self) -> int:
    """Returns the pad id."""
    return self.vocab["<pad>"]

  def __len__(self) -> int:
    """Returns the total vocabulary size."""
    return self.tokenizer.get_vocab_size()


class CharacterVocab(EncoderVocab[str]):
  """Character-level vocabulary: each character is a single token.

  Tokens are sorted by character code, with <pad>=0 and <unk>=1.
  """

  # All printable ASCII characters (0x20-0x7E) plus tab and newline.
  DEFAULT_CHARS: str = (
      " \t\n!\"#$%&'()*+,-./"
      "0123456789:;<=>?@"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
      "abcdefghijklmnopqrstuvwxyz{|}~"
  )

  def __init__(self, chars: str | None = None):
    """Initializes the character vocabulary.

    Args:
      chars: Characters to include. Defaults to printable ASCII + tab/newline.
        Duplicate characters are silently deduplicated.
    """
    specials = ["<pad>", "<unk>"]
    unique_chars = sorted(set(self.DEFAULT_CHARS if chars is None else chars))

    self._stoi: dict[str, int] = {}
    for i, s in enumerate(specials):
      self._stoi[s] = i
    for i, c in enumerate(unique_chars):
      self._stoi[c] = i + len(specials)

  def to_token_ids(self, obj: str, /) -> list[int]:
    """Converts a string to a list of character-level token ids."""
    return [self._stoi.get(c, self._stoi["<unk>"]) for c in obj]

  @property
  def pad_id(self) -> int:
    return self._stoi["<pad>"]

  def __len__(self) -> int:
    return len(self._stoi)


open_file = open


class SentencePieceVocab(EncoderVocab[str]):
  """SentencePiece vocab."""

  T5_FILE = "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"

  def __init__(self, file_path: str):
    """Initializes SentencePieceVocab by loading a pre-trained .model file."""

    if file_path.startswith('gs://'):  # Check Google Cloud Storage path.
      import gcsfs, os

      local_path = f'/tmp/{os.path.basename(file_path)}'
      gcsfs.GCSFileSystem(token='anon').get(file_path, local_path)
      file_path = local_path

    self.file_path = file_path
    self.sp_processor = spp.SentencePieceProcessor()
    self.sp_processor.Load(self.file_path)

    if self.sp_processor.pad_id() == -1:
      raise ValueError(
          f"SentencePiece model '{file_path}' does not have a PAD token"
          " explicitly defined."
      )

  # Helps multiprocessing pickling, which fails with C++ SentencePieceProcessor.
  def __getstate__(self) -> dict[str, Any]:
    with open_file(self.file_path, "rb") as f:
      return {"model_blob": f.read()}

  def __setstate__(self, state: dict[str, Any]) -> None:
    self.sp_processor = spp.SentencePieceProcessor()
    self.sp_processor.LoadFromSerializedProto(state["model_blob"])

  def to_token_ids(self, obj: str, /) -> list[int]:
    """Converts text to a list of token ids using the SentencePiece model."""
    return self.sp_processor.EncodeAsIds(obj)

  @property
  def pad_id(self) -> int:
    """Returns the pad id defined in the SentencePiece model."""
    return self.sp_processor.pad_id()

  def __len__(self) -> int:
    """Returns the total vocabulary size."""
    return self.sp_processor.GetPieceSize()

  @classmethod
  def from_t5(cls) -> "SentencePieceVocab":
    return cls(cls.T5_FILE)

  @classmethod
  def from_corpus(
      cls,
      corpus_path: str | pathlib.Path | list[str | pathlib.Path],
      vocab_size: int = 8192,
      model_prefix: str | pathlib.Path | None = None,
      sentencepiece_trainer_kwargs: dict[str, str] | None = None,
  ) -> "SentencePieceVocab":
    """Trains a SentencePiece vocab from the given corpus."""
    if model_prefix is None:
      model_prefix = pathlib.Path("/tmp/trained_sentencepiece")

    if isinstance(corpus_path, list):
      corpus_path = ",".join(map(str, corpus_path))

    trainer_args = {
        "input": str(corpus_path),
        "model_prefix": str(model_prefix),
        "vocab_size": str(vocab_size),
        "model_type": "bpe",
        "pad_id": "0",
        "unk_id": "1",
        "pad_piece": "<pad>",
        "unk_piece": "<unk>",
        "bos_id": "-1",
        "eos_id": "-1",
        "hard_vocab_limit": "false",
        "byte_fallback": "true",
        "split_by_number": "true",
        "split_digits": "true",
        "split_by_unicode_script": "false",
        "character_coverage": "1.0",
        "input_sentence_size": "0",
        "max_sentence_length": "500000",
        "shuffle_input_sentence": "false",
        "num_threads": "1",
    }
    if sentencepiece_trainer_kwargs:
      trainer_args.update(sentencepiece_trainer_kwargs)
    cmd = " ".join(f"--{k}={v}" for k, v in trainer_args.items())
    spt.SentencePieceTrainer.Train(cmd)
    return cls(str(model_prefix) + ".model")


class HuggingFaceVocab(EncoderVocab[str]):
  """An EncoderVocab that wraps HuggingFace."""

  def __init__(self, model_name: str, **tokenizer_kwargs):
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, **tokenizer_kwargs
    )

  def to_token_ids(self, obj: str, /) -> list[int]:
    return self.tokenizer.encode(obj)

  @property
  def pad_id(self) -> int:
    return self.tokenizer.pad_token_id

  def __len__(self) -> int:
    return len(self.tokenizer)


class XValVocabWrapper(EncoderVocab[str]):
  """Wraps any EncoderVocab to add xVal continuous number encoding.

  Each number in the input text is replaced with a single <num> token.
  The raw float value is preserved in per-token features (`is_float`,
  `float_value`) for the embedder to interpret.

  The vocab adds exactly one extra token ID beyond the base vocab. The
  scale decomposition (powers of 10, tanh, etc.) is handled entirely
  by the embedder, not here.
  """

  _NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

  def __init__(self, base_vocab: EncoderVocab[str]):
    self.base_vocab = base_vocab
    self._num_token_id = len(base_vocab)  # Single <num> ID.

  @property
  def features_spec(self) -> dict[str, FeatureSpec]:
    return {
        "ids": FeatureSpec(dtype=int, padding=self.pad_id),
        "is_float": FeatureSpec(dtype=bool, padding=False),
        "float_value": FeatureSpec(dtype=float, padding=0.0),
    }

  def _append_text(self, text: str, ids, is_float, float_value):
    """Tokenizes text via base vocab and appends non-float features."""
    text_ids = self.base_vocab.to_token_ids(text)
    ids.extend(text_ids)
    is_float.extend([False] * len(text_ids))
    float_value.extend([0.0] * len(text_ids))

  def to_features(self, text: str, /) -> dict[str, list]:
    """Tokenizes text, replacing each number with one <num> token."""
    ids, is_float, float_value = [], [], []
    last = 0
    for m in self._NUMBER_RE.finditer(text):
      if m.start() > last:
        self._append_text(text[last : m.start()], ids, is_float, float_value)
      ids.append(self._num_token_id)
      is_float.append(True)
      float_value.append(float(m.group()))
      last = m.end()
    if last < len(text):
      self._append_text(text[last:], ids, is_float, float_value)
    return {"ids": ids, "is_float": is_float, "float_value": float_value}

  def to_token_ids(self, text: str, /) -> list[int]:
    return self.to_features(text)["ids"]

  @property
  def pad_id(self) -> int:
    return self.base_vocab.pad_id

  def __len__(self) -> int:
    return len(self.base_vocab) + 1  # base + <num>
