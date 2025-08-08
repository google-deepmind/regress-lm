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

from regress_lm.models.pytorch import encoders
import torch
from absl.testing import absltest


class MambaEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.d_model = 16
    self.encoder = encoders.MambaEncoder(d_model=self.d_model, num_layers=1)

  def test_forward(self):
    batch_size, seq_len = 4, 50

    src_emb = torch.randn(batch_size, seq_len, self.d_model)
    output = self.encoder(src_emb, src_key_padding_mask=None)
    self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))


if __name__ == "__main__":
  absltest.main()
