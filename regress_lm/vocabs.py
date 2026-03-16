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
import pathlib
from typing import Any, Generic, Sequence, TypeVar
from regress_lm import tokenizers
import tokenizers as ht
import transformers
import sentencepiece as spp
import sentencepiece as spt
import torch


ObjectT = TypeVar("ObjectT")


class BaseVocab(abc.ABC, Generic[ObjectT]):
  """Base class for vocabularies."""

  @abc.abstractmethod
  def to_token_ids(self, obj: ObjectT, /) -> list[int]:
    """Converts object (e.g. text) to token ids."""

  @abc.abstractmethod
  def __len__(self) -> int:
    """Returns the vocab size."""


class EncoderVocab(BaseVocab[ObjectT]):
  """Vocabulary class for encoders.

  Note we don't ever need to convert back to text.
  """

  @property
  @abc.abstractmethod
  def pad_id(self) -> int:
    """Returns the pad id."""


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

  def get_allowed_tokens_mask(
      self, prev_tokens: torch.Tensor, device: torch.device
  ) -> torch.Tensor:
    """Returns a boolean mask of allowed next tokens for a batch.

    Args:
      prev_tokens: (batch_size, seq_len) tensor of previous token IDs.
      device: The device to create the mask on.

    Returns:
      (batch_size, vocab_size) boolean tensor where True means allowed.
    """
    batch_size, seq_len = prev_tokens.shape
    obj_step = seq_len % self.num_tokens_per_obj

    # 1. Base mask based on position in the object (0, 1, ... num_tokens_per_obj-1)
    # We cache these to avoid recomputing.
    if not hasattr(self, "_cached_pos_masks"):
      self._cached_pos_masks = {}

    if obj_step not in self._cached_pos_masks:
      # Create a dummy sequence to query the tokenizer
      # We just need the length to be correct for standard tokenizers
      dummy_prev = [self.pad_token] * obj_step  # Content doesn't matter for pos-based
      # Note: This assumes standard tokenizers only care about length/position
      # for the "structure" of the float.
      # Special values are handled separately below.
      
      # We need to be careful: possible_next_tokens might raise if we pass garbage
      # but for P10/IEEE/Normalized, they check length.
      # Let's try to construct a valid-looking prefix if possible, or just rely on
      # the fact that they check length.
      # Actually, let's use the tokenizer's logic directly if possible, or just
      # use the existing possible_next_tokens with a "safe" prefix.
      
      # For P10: checks index.
      # For IEEE: checks index.
      # For Normalized: checks index.
      # For AddSpecialValues: checks index AND content of first token.
      
      # So for the "base" mask, we want the mask assuming it's a "normal" number.
      # We can pass a prefix that indicates a normal number.
      # e.g. ["+"] if obj_step=1.
      
      safe_prefix = []
      if obj_step > 0:
         # Try to find a "normal" token to start with.
         # For P10/IEEE, '+' is usually safe.
         # We can iterate self.itos and find one that isn't special.
         # But simpler: just use the first token that is valid for step 0.
         pass # We will handle this by just using the existing method with a "clean" prefix
      
      # Actually, we can just compute the mask for a "generic" case.
      # But wait, AddSpecialValues *wraps* another tokenizer.
      # If we ask AddSpecialValues.possible_next_tokens with a "normal" prefix,
      # it delegates to the inner tokenizer. That's exactly what we want for the base mask.
      
      # Construct a prefix that looks "normal".
      # We can just use the first valid token for each step to build a chain.
      # This is a bit hacky but robust enough for these specific tokenizers.
      prefix = []
      for _ in range(obj_step):
          allowed = self.possible_next_token_ids([self.stoi[t] for t in prefix])
          # Pick the first one
          prefix.append(self.itos[allowed[0]])
      
      allowed_indices = self.possible_next_token_ids([self.stoi[t] for t in prefix])
      mask = torch.zeros(len(self), dtype=torch.bool, device=device)
      mask[allowed_indices] = True
      self._cached_pos_masks[obj_step] = mask

    # Start with the cached mask for this position
    mask = self._cached_pos_masks[obj_step].to(device).repeat(batch_size, 1)

    # 2. Handle Special Values (NAN, INF, INVALID)
    # If we are inside an object (obj_step > 0), we must check if we started with a special token.
    if obj_step > 0:
      # Check if the tokenizer has special values
      # We look for the AddSpecialValues wrapper or similar logic.
      # We can detect this by checking if there are tokens that force themselves to be repeated.
      # Or more explicitly, we can check if the tokenizer has `_special_tokens`.
      
      # Let's look at the tokenizer chain.
      tokenizer = self.tokenizer
      special_tokens_map = {} # start_token_id -> special_token_id (usually same)
      
      # Unwrap to find AddSpecialValues
      curr = tokenizer
      while hasattr(curr, "tokenizer") or hasattr(curr, "_tokenizer"):
          if hasattr(curr, "_special_tokens"):
             # Found it.
             for st in curr._special_tokens:
                 if st in self.stoi:
                     sid = self.stoi[st]
                     special_tokens_map[sid] = sid
             break
          curr = getattr(curr, "tokenizer", getattr(curr, "_tokenizer", None))

      if special_tokens_map:
          # Find which sequences started with a special token
          # The start of the current object is at index: seq_len - obj_step
          start_token_ids = prev_tokens[:, seq_len - obj_step]
          
          for special_id in special_tokens_map:
              # Identify rows where the object started with this special token
              is_special = (start_token_ids == special_id)
              
              if is_special.any():
                  # For these rows, the ONLY allowed token is the special token itself
                  # (assuming special values are repeated like <NAN><NAN><NAN>)
                  # We zero out the mask for these rows and set the special token to True
                  mask[is_special] = False
                  mask[is_special, special_id] = True

    return mask

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
        "split_by_number": "false",
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
