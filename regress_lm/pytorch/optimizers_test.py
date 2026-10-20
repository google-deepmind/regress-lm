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

"""Tests for custom PyTorch optimizers."""

from typing import cast

from regress_lm import rlm
from regress_lm import vocabs
from regress_lm.pytorch import optimizers
from torch import nn
from torch import optim

from absl.testing import absltest


def _param_names(optimizer: optim.Optimizer, names: dict[int, str]) -> set[str]:
  out = set()
  for group in optimizer.param_groups:
    out.update(names[id(p)] for p in group["params"])
  return out


class MuonAdamWTest(absltest.TestCase):

  def test_param_routing(self):
    reg_lm = rlm.RegressLM.from_scratch(
        encoder_vocab=vocabs.BasicEnglishVocab(["hello", "world"]),
        max_input_len=16,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    named_params = list(cast(nn.Module, reg_lm.model).named_parameters())
    names = {id(p): name for name, p in named_params}
    opt = optimizers.muon_adamw(named_params)
    muon = _param_names(opt.optimizer_1, names)
    adamw = _param_names(opt.optimizer_2, names)

    # Muon gets exactly the hidden 2D attention and FFN weights.
    self.assertEqual(
        muon,
        {
            "encoder_decoder.encoder.layers.0.q_proj.weight",
            "encoder_decoder.encoder.layers.0.k_proj.weight",
            "encoder_decoder.encoder.layers.0.v_proj.weight",
            "encoder_decoder.encoder.layers.0.out_proj.weight",
            "encoder_decoder.encoder.layers.0.linear1.weight",
            "encoder_decoder.encoder.layers.0.linear2.weight",
            "encoder_decoder.decoder.layers.0.self_attn.q_proj.weight",
            "encoder_decoder.decoder.layers.0.self_attn.k_proj.weight",
            "encoder_decoder.decoder.layers.0.self_attn.v_proj.weight",
            "encoder_decoder.decoder.layers.0.self_attn.out_proj.weight",
            "encoder_decoder.decoder.layers.0.cross_attn.q_proj.weight",
            "encoder_decoder.decoder.layers.0.cross_attn.k_proj.weight",
            "encoder_decoder.decoder.layers.0.cross_attn.v_proj.weight",
            "encoder_decoder.decoder.layers.0.cross_attn.out_proj.weight",
            "encoder_decoder.decoder.layers.0.linear1.weight",
            "encoder_decoder.decoder.layers.0.linear2.weight",
        },
    )
    # AdamW gets everything else; its only 2D weights are the encoder and
    # decoder token embeddings and the output head.
    self.assertEqual(adamw, set(names.values()) - muon)
    self.assertEqual(
        {name for name, p in named_params if p.ndim == 2} - muon,
        {
            "encoder_decoder.encoder.embedding.weight",
            "encoder_decoder.tgt_tok_emb.weight",
            "encoder_decoder.generator.weight",
        },
    )


if __name__ == "__main__":
  absltest.main()
