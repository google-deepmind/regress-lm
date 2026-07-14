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

import numpy as np
from unittest import mock
from regress_lm import core
from regress_lm import rlm
from absl.testing import absltest


class RlmTest(absltest.TestCase):

  def test_demo(self):
    reg_lm = rlm.RegressLM.from_scratch(
        max_input_len=128,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        compile_model=False,
    )
    examples = [core.Example(x='hello', y=0.3), core.Example(x='world', y=0.7)]
    reg_lm.fine_tune(examples)

    query1, query2 = core.ExampleInput(x='hi'), core.ExampleInput(x='bye')
    samples1, samples2 = reg_lm.sample([query1, query2], num_samples=128)
    self.assertLen(samples1, 128)
    self.assertLen(samples2, 128)

  def test_sample_forwards_temperature(self):
    mock_model = mock.MagicMock()
    mock_model.converter.convert_inputs.return_value = {'encoder_input': object()}
    mock_model.decode.return_value = (
        None,
        np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32),
    )
    reg_lm = rlm.RegressLM(mock_model)

    output = reg_lm.sample(
        [core.ExampleInput(x='hello')], num_samples=3, temperature=0.5
    )

    self.assertLen(output, 1)
    np.testing.assert_array_equal(
      output[0], np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    )
    mock_model.decode.assert_called_once_with(
        {'encoder_input': mock.ANY}, 3, temperature=0.5
    )

  def test_sample_uses_default_temperature(self):
    mock_model = mock.MagicMock()
    mock_model.converter.convert_inputs.return_value = {'encoder_input': object()}
    mock_model.decode.return_value = (
        None,
        np.array([[[1.0], [2.0]]], dtype=np.float32),
    )
    reg_lm = rlm.RegressLM(mock_model)

    reg_lm.sample([core.ExampleInput(x='hello')], num_samples=2)

    mock_model.decode.assert_called_once_with(
        {'encoder_input': mock.ANY}, 2, temperature=1.0
    )


if __name__ == '__main__':
  absltest.main()
