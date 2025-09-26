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

"""Example single-process pretraining script for RegressLM."""

import collections
import math
from typing import Literal, Sequence

from absl import app
from absl import flags
from clu import metric_writers
import numpy as np
from regress_lm import core
from regress_lm.pytorch import data_utils
from regress_lm.pytorch import model as pytorch_model
import scipy.stats
import torch
from torch import optim
from torch import utils
from torch.optim import lr_scheduler


# Training
flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
flags.DEFINE_integer("num_epochs", 100, "Number of training epochs.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_float("warmup_steps_fraction", 0.1, "Fraction of warmup steps.")

# Validation
flags.DEFINE_integer("validate_every_steps", 500, "Validation frequency.")
flags.DEFINE_integer(
    "validation_batch_size", 16, "Internal batch size for inference."
)

# Final inference
flags.DEFINE_integer(
    "inference_batch_size", 16, "Internal batch size for inference."
)
flags.DEFINE_integer("num_samples", 1024, "Number of samples for prediction.")


FLAGS = flags.FLAGS


def load_examples(
    split: Literal["train", "validation", "test"],
) -> list[core.Example]:
  del split
  raise NotImplementedError("User must implement this.")


def get_model() -> pytorch_model.PyTorchModel:
  raise NotImplementedError("User must implement this.")


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str
) -> dict[str, float]:
  return {
      f"{prefix}_mse": float(np.mean((y_true - y_pred) ** 2)),
      f"{prefix}_pearson": scipy.stats.pearsonr(y_true, y_pred).correlation,
      f"{prefix}_kendall": scipy.stats.kendalltau(y_true, y_pred).correlation,
      f"{prefix}_spearman": scipy.stats.spearmanr(y_true, y_pred).correlation,
  }


def get_prediction_metrics(
    model: pytorch_model.PyTorchModel,
    examples: Sequence[core.Example],
    *,
    num_samples: int,
    inference_batch_size: int,
) -> dict[str, float]:
  """Run autoregressive sampling and return per-example predictions."""
  model.eval()
  all_preds: list[np.ndarray] = []

  # Create a dataset and dataloader for inference.
  inference_dl = utils.data.DataLoader(
      dataset=data_utils.ExampleDataset(examples),
      batch_size=inference_batch_size,
      shuffle=False,
      collate_fn=model.convert_examples,
  )

  with torch.no_grad():
    for batch in inference_dl:
      _, floats = model.decode(batch, num_samples=num_samples)
      all_preds.append(floats)

  preds = np.concatenate(all_preds, axis=0)
  true_y = np.array([ex.y for ex in examples])
  res_mean = regression_metrics(true_y, np.mean(preds, axis=1), "mean")
  res_med = regression_metrics(true_y, np.median(preds, axis=1), "median")
  return {**res_mean, **res_med}


def make_scheduler(
    optimiser: optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warm_frac: float,
) -> lr_scheduler.LambdaLR:
  """Creates a learning rate scheduler."""
  total = steps_per_epoch * epochs
  warm = int(total * warm_frac)

  def lr_lambda(step: int) -> float:
    if step < warm:
      return step / max(1, warm)
    progress = (step - warm) / max(1, total - warm)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

  return lr_scheduler.LambdaLR(optimiser, lr_lambda)


def main(_) -> None:
  model = get_model()
  optimizer = optim.Adafactor(model.parameters(), lr=FLAGS.learning_rate)

  writer = metric_writers.create_default_writer(
      write_to_xm_measurements=True, write_to_datatable=True, asynchronous=False
  )

  # Streaming dataloader.
  train_dl = utils.data.DataLoader(
      dataset=data_utils.ExampleDataset(load_examples("train")),
      batch_size=FLAGS.batch_size,
      shuffle=True,
      num_workers=16,
      pin_memory=True,
      persistent_workers=True,
      drop_last=False,
      collate_fn=model.convert_examples,
  )
  val_dl = utils.data.DataLoader(
      dataset=data_utils.ExampleDataset(load_examples("validation")),
      batch_size=FLAGS.validation_batch_size,
      shuffle=False,
      num_workers=4,  # Can use fewer workers for validation
      collate_fn=model.convert_examples,
  )

  steps_per_epoch = math.ceil(len(train_dl) / FLAGS.batch_size)
  scheduler = make_scheduler(
      optimizer, steps_per_epoch, FLAGS.num_epochs, FLAGS.warmup_steps_fraction
  )

  global_step = 0
  for _ in range(FLAGS.num_epochs):
    model.train()
    for batch in train_dl:
      global_step += 1

      optimizer.zero_grad(set_to_none=True)
      losses_per_example, metrics = model.compute_losses_and_metrics(batch)
      loss = losses_per_example.mean()
      loss.backward()

      optimizer.step()
      scheduler.step()

      metrics = {f"train_{k}": v.item() for k, v in metrics.items()}
      metrics["train_loss"] = loss.item()
      writer.write_scalars(global_step, metrics)

      # Validation
      if global_step % FLAGS.validate_every_steps == 0:
        model.eval()

        total_loss, total_items = 0.0, 0
        metric_sums: dict[str, float] = collections.defaultdict(float)

        with torch.no_grad():
          for val_batch in val_dl:
            losses_p_ex, metrics = model.compute_losses_and_metrics(val_batch)
            bsz = next(iter(val_batch.values())).size(0)

            total_loss += losses_p_ex.sum().item()
            for k, m in metrics.items():
              metric_sums[k] += m.item() * bsz
            total_items += bsz

        # Calculate final metrics
        val_metrics = {
            f"validation_{k}": v / total_items for k, v in metric_sums.items()
        }
        val_metrics["validation_loss"] = total_loss / total_items
        writer.write_scalars(global_step, val_metrics)
        # Set model back to training mode
        model.train()

  # ------------ Final test evaluation ------------
  test_metrics = get_prediction_metrics(
      model,
      examples=load_examples("test"),
      num_samples=FLAGS.num_samples,
      inference_batch_size=FLAGS.inference_batch_size,
  )
  writer.write_scalars(global_step, test_metrics)


if __name__ == "__main__":
  app.run(main)
