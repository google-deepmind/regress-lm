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

from typing import Any

from regress_lm import core
from regress_lm import tokenizers
from regress_lm import vocabs
from regress_lm.pytorch import data_utils
from regress_lm.pytorch import model as model_lib
from regress_lm.pytorch import training
import torch
from torch import optim
from torch.optim import lr_scheduler

from absl.testing import absltest


class TrainingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.encoder_vocab = vocabs.BasicEnglishVocab(['hello', 'world'])
    self.decoder_tokenizer = tokenizers.P10Tokenizer()
    self.decoder_vocab = vocabs.DecoderVocab(self.decoder_tokenizer)
    self.architecture_kwargs = dict(
        d_model=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    self.cfg = model_lib.PyTorchModelConfig(
        encoder_vocab=self.encoder_vocab,
        decoder_vocab=self.decoder_vocab,
        max_input_len=4,
        architecture_kwargs=self.architecture_kwargs,
    )
    self.model = self.cfg.make_model()

    ds = data_utils.ExampleDataset([
        core.Example(x='hello', y=1.0),
        core.Example(x='world', y=2.0),
        core.Example(x='good', y=3.0),
        core.Example(x='bye', y=4.0),
    ])
    self.trainer = training.Trainer(
        model=self.model,
        optimizer_factory=optim.Adafactor,
        scheduler_factory=lr_scheduler.ConstantLR,
        train_ds=ds,
        batch_size=2,
        use_ddp=False,  # Can't test distributed training.
        num_data_workers=2,
        compile_model=False,  # No compiler for CPU case.
    )

  def test_train_and_validation_smoke(self):
    for batch in self.trainer.train_dl:
      self.trainer.run_train_step(batch)

    self.trainer.run_eval_epoch(self.trainer.train_dl)

  def _save(self) -> str:
    path = self.create_tempfile().full_path
    self.trainer.save_checkpoint(path)
    self.trainer._ckpt_thread.join()  # pylint: disable=protected-access
    return path

  def _raw_and_ema_states(self) -> tuple[Any, Any]:
    raw = {k: v.clone() for k, v in self.model.state_dict().items()}
    with self.trainer.ema_parameters():
      ema = {k: v.clone() for k, v in self.model.state_dict().items()}
    return raw, ema

  def test_ema(self):
    old_path = self._save()  # No EMA updates yet, as in older checkpoints.
    batch = next(iter(self.trainer.train_dl))
    for _ in range(2):  # EMA is on by default.
      self.trainer.run_train_step(batch)
    raw, ema = self._raw_and_ema_states()
    self.assertFalse(all(torch.equal(ema[k], raw[k]) for k in raw))
    self.trainer.run_eval_epoch(self.trainer.train_dl)
    torch.testing.assert_close(self.model.state_dict(), raw)  # Restored.

    path = self._save()
    self.trainer.run_train_step(batch)
    expected = self._raw_and_ema_states()
    ckpt = self.trainer.load_checkpoint(path)
    torch.testing.assert_close(ckpt['model_state'], ema)  # Read downstream.
    torch.testing.assert_close(self._raw_and_ema_states(), (raw, ema))
    self.trainer.run_train_step(batch)  # Resumes exactly.
    torch.testing.assert_close(self._raw_and_ema_states(), expected)

    ckpt = self.trainer.load_checkpoint(old_path)
    self.assertNotIn('train_model_state', ckpt)
    for state in self._raw_and_ema_states():  # EMA is a no-op again.
      torch.testing.assert_close(state, ckpt['model_state'])


if __name__ == '__main__':
  absltest_launcher.main()
