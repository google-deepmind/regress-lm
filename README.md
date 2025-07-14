# RegressLM: Easy Text-to-Text Regression
[![Continuous Integration](https://github.com/google-deepmind/regress-lm/actions/workflows/core_test.yml/badge.svg)](https://github.com/google-deepmind/regress-lm/actions?query=branch%3Amain)

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

## Usage
There are two main stages: **inference** and **pretraining** (optional).

## Inference
The intended use-case is to import a RegressLM class, which can decode
floating-point predictions from a given input, and also fine-tune against new
data.

```python
from regress_lm import core
from regress_lm import rlm

# Create RegressLM with max input token length.
reg_lm = rlm.RegressLM.from_default(max_input_len=2048)

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
from regress_lm.models.pytorch import model as torch_model_lib

model = torch_model_lib.PyTorchModel(...)
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

## Boosting Performance
Below, we describe many ways to improve performance.

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

## Contributors and Citation
The codebase was written by: Xingyou Song, Yash Akhauri, Dara Bahri, Michal
Lukasik, Arissa Wongpanich, Adrian N. Reyes, and Bryan Lewandowski.

If you find this project useful, please consider citing our work as follows:

```
@article{akhauri2025performancepredictionlargesystems,
      title={Performance Prediction for Large Systems via Text-to-Text Regression},
      author={Yash Akhauri and Bryan Lewandowski and Cheng-Hsi Lin and Adrian N. Reyes and Grant C. Forbes and Arissa Wongpanich and Bangding Yang and Mohamed S. Abdelfattah and Sagi Perel and Xingyou Song},
      journal={arXiv preprint arXiv:2506.21718},
      year={2025}
}
```

**Disclaimer:** This is not an officially supported Google product.