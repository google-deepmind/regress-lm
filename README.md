# RegressLM: Easy Text-to-Text Regression
[![Continuous Integration](https://github.com/google-deepmind/regress-lm/actions/workflows/core_test.yml/badge.svg)](https://github.com/google-deepmind/regress-lm/actions?query=branch%3Amain)

  [**Google Research Blog**](https://research.google/blog/simulating-large-systems-with-regression-language-models/)
| [**Setup**](#setup)
| [**Usage**](#usage)
| [**Extended Usage**](#extended_usage)
| [**Citing**](#citing)

## Overview
RegressLM is a library for text-to-text regression, applicable to any input
string representation and allows pretraining and fine-tuning over multiple
regression tasks.

<figure>
<p align="center" width=65%>
<img src="https://raw.githubusercontent.com/akhauriyash/figures_placeholder/refs/heads/main/teaser_rlm_compressed.gif" alt="RegressLM decoding a numerical performance metric from text."/>
  <br>
  <figcaption style="text-align: center;"><em><b><a href="https://arxiv.org/abs/2506.21718">Example Application</a>: Directly regressing performance metrics from unstructured, textually represented system states from Google's massive compute clusters.</b></em></figcaption>
</p>
</figure>

## Setup <a name="setup"></a>
Get started by installing the core libraries:

```
pip install -e .
```

To run e.g. T5Gemma variants, install additional libraries:

```
pip install ".[extras]"
```

## Usage <a name="usage"></a>
There are two main stages: **inference** and **pretraining** (optional).

## Inference
The intended use-case is to import a RegressLM class, which can decode
floating-point predictions from a given input, and also fine-tune against new
data.

```python
from regress_lm import core
from regress_lm import rlm

# Create RegressLM from scratch. Optionally, use `from_t5gemma_encoder`.
reg_lm = rlm.RegressLM.from_scratch(max_input_len=2048)

# Example (x,y) pairs, which can be fine-tuned against.
examples = [core.Example(x='hello', y=0.3), core.Example(x='world', y=-0.3)]
reg_lm.fine_tune(examples)

# Query inputs.
query1, query2 = core.ExampleInput(x='hi'), core.ExampleInput(x='bye')
samples1, samples2 = reg_lm.sample([query1, query2], num_samples=128)
```

## Pretraining
To produce better initial checkpoints for transfer learning, we recommend
the user pretrains over large amounts of their own training data. Example
pseudocode with PyTorch:

```python
from torch import optim
from regress_lm.pytorch import model as model_lib

model = model_lib.PyTorchModel(...)
optimizer = optim.Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)
for _ in range(...):
  examples = [Example(x=..., y=...), ...]
  tensor_examples = model.convert(examples)
  optimizer.zero_grad()
  loss, _ = model.compute_loss_and_metrics(tensor_examples)
  loss.backward()
  optimizer.step()
```

## Boosting Performance and Extended Applications <a name="extended_usage"></a>
Below, we describe ways to improve performance and extended applications, using
lower level API.

### Train Custom Vocabulary
You can generate a custom vocabulary, trained on an offline corpus of data
`mydata.txt`:

```python
encoder_vocab = SentencePieceVocab.from_corpus(corpus_path='mydata.txt', vocab_size=1024)
```

### Larger Sizes
Larger model sizes may increase performance, although with more computational
cost:

```python
model = PyTorchModel(num_encoder_layers=12, num_decoder_layers=12)
```

### Multi-objective Support
The RLM can decode a concatenated sequence of tokens too, for multi-objective
regression:

```python
reg_lm = rlm.RegressLM.from_scratch(max_num_objs=2)

# Examples can have variable objective lengths.
examples = [core.Example(x='hello', y=[0.2]), core.Example(x='world', y=[-0.2, 0.3])]
reg_lm.fine_tune(examples)

# Now `samples` has shape (128, 2).
samples = reg_lm.sample([core.ExampleInput(x='hi')], num_samples=128)[0]
```

### Pretrained Third-Party Models
[T5Gemma](https://developers.googleblog.com/en/t5gemma/) frozen encoder + our
default decoder is supported:

```python
model = PyTorchModel(encoder_type=EncoderType.T5GEMMA)
```

End-to-end T5Gemma is also supported:

```python
from regress_lm.pytorch import t5gemma_model
model = t5gemma_model.T5GemmaModel('google/t5gemma-s-s-prefixlm')
```

### Long-Context
To support 100K+ input token lengths, alternative encoders (e.g.
[`mamba-ssm`](https://github.com/state-spaces/mamba) and [Performer](https://research.google/blog/rethinking-attention-with-performers/)) are supported:

```python
model = PyTorchModel(encoder_type=EncoderType.MAMBA, additional_encoder_kwargs={'d_state': 128})
model = PyTorchModel(encoder_type=EncoderType.PERFORMER, additional_encoder_kwargs={'num_features': 256})
```

## Contributors and Citation <a name="citing"></a>
The codebase was written by: Xingyou Song, Yash Akhauri, Dara Bahri, Michal
Lukasik, Arissa Wongpanich, Adrian N. Reyes, and Bryan Lewandowski.

If you find this project useful, please consider citing the relevant works:

```
@article{performance_prediction,
      title={Performance Prediction for Large Systems via Text-to-Text Regression},
      author={Yash Akhauri and Bryan Lewandowski and Cheng-Hsi Lin and Adrian N. Reyes and Grant C. Forbes and Arissa Wongpanich and Bangding Yang and Mohamed S. Abdelfattah and Sagi Perel and Xingyou Song},
      journal={arXiv preprint arXiv:2506.21718},
      year={2025}
}

@article{omnipred,
      title={OmniPred: Language Models as Universal Regressors},
      author={Xingyou Song and Oscar Li and Chansoo Lee and Bangding Yang and Daiyi Peng and Sagi Perel and Yutian Chen},
      journal={Trans. Mach. Learn. Res.},
      year={2024},
      url={https://openreview.net/forum?id=t9c3pfrR1X},
}

@article{decoding_regression,
      title={Decoding-based Regression},
      author={Xingyou Song and Dara Bahri},
      journal={Trans. Mach. Learn. Res.},
      year={2025},
      url={https://openreview.net/forum?id=avUQ8jguxg},
}
```

**Disclaimer:** This is not an officially supported Google product.