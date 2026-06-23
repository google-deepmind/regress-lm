# RegressLM: Easy Sequence-to-Sequence Numeric Prediction
[![Continuous Integration](https://github.com/google-deepmind/regress-lm/actions/workflows/core_test.yml/badge.svg)](https://github.com/google-deepmind/regress-lm/actions?query=branch%3Amain)

  [**Overview**](#overview)
| [**Setup**](#setup)
| [**Usage**](#usage)
| [**Extended Usage**](#extended_usage)

**Core Contributors**: Xingyou Song, Yash Akhauri, Jiyoun Ha, Bryan Lewandowski

## Overview <a name="overview"></a>
RegressLM is a library for sequence-to-sequence numeric prediction, applicable
to tokenizable inputs (e.g. strings, images) and allows pretraining and
fine-tuning over multiple tasks.

<p align="center">
<img src="https://raw.githubusercontent.com/akhauriyash/figures_placeholder/refs/heads/main/teaser_rlm_compressed.gif" alt="RegressLM decoding a numerical performance metric from text." width="100%"/>
  <br>
  <em><b><a href="https://research.google/blog/simulating-large-systems-with-regression-language-models/">Example Application</a>: Directly predicting performance metrics from unstructured, textually represented system states from Google's massive compute clusters.</b></em>
</p>

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
There are two main stages: **inference** and **pretraining** (optional but
recommended).

### Inference
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

### Pretraining
To produce better initial checkpoints for transfer learning, we recommend
the user pretrains over large amounts of their own training data. Example
pseudocode with PyTorch:

```python
from regress_lm.pytorch import model as model_lib
from regress_lm.pytorch import training

model = model_lib.PyTorchModelConfig(...).make_model()
trainer = training.Trainer(model, optimizer_factory, train_dataset, ...)
for batch in trainer.train_dl:
  train_metrics = trainer.run_train_step(batch)
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
config = PyTorchModelConfig(architecture_kwargs=dict(num_encoder_layers=12, num_decoder_layers=12))
```

### Multi-objective Support
The RLM can decode a concatenated sequence of tokens too, for multi-objective
prediction:

```python
reg_lm = rlm.RegressLM.from_scratch(max_num_objs=2)

# Examples can have variable objective lengths.
examples = [core.Example(x='hello', y=[0.2]), core.Example(x='world', y=[-0.2, 0.3])]
reg_lm.fine_tune(examples)

# Now `samples` has shape (128, 2).
samples = reg_lm.sample([core.ExampleInput(x='hi')], num_samples=128)[0]
```

### Pretrained Third-Party Models
T5Gemma ([V1](https://developers.googleblog.com/en/t5gemma/) + [V2](https://blog.google/innovation-and-ai/technology/developers-tools/t5gemma-2/)) encoder + our
custom decoder is supported:

```python
config = PyTorchModelConfig(architecture_kwargs=dict(encoder_type=EncoderType.T5GEMMA))
```

End-to-end T5Gemma (encoder + decoder) is also supported as a baseline:

```python
from regress_lm.pytorch import t5gemma_model
model = t5gemma_model.T5GemmaModelConfig('google/t5gemma-s-s-prefixlm').make_model()
```

### Long-Context
To support 100K+ input token lengths, alternative encoders (e.g.
[`mamba-ssm`](https://github.com/state-spaces/mamba) and [Performer](https://research.google/blog/rethinking-attention-with-performers/)) are supported:

```python
architecture_kwargs = dict(encoder_type=EncoderType.MAMBA, additional_encoder_kwargs={'d_state': 128})
architecture_kwargs = dict(encoder_type=EncoderType.PERFORMER, additional_encoder_kwargs={'num_features': 256})
```

**Disclaimer:** This is not an officially supported Google product.