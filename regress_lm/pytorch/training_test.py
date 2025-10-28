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

from regress_lm import core
from regress_lm import tokenizers
from regress_lm import vocabs
from regress_lm.pytorch import data_utils
from regress_lm.pytorch import model as model_lib
from regress_lm.pytorch import training
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
    self.model = self.cfg.make_model(compile_model=False)

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
        validation_ds=ds,
        batch_size=2,
        use_ddp=False,  # Can't test distributed training.
        num_data_workers=2,
    )

  def test_train_and_validation_smoke(self):
    for batch in self.trainer.train_dl:
      self.trainer.run_train_step(batch)

    self.trainer.run_validation_epoch()


if __name__ == '__main__':
  absltest_launcher.main()
