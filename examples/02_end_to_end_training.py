import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import optim
from regress_lm import core
from regress_lm.pytorch import model as model_lib

def main(args):
    # Load and prepare data
    data = pd.read_csv(args.dataset_path)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=args.seed)

    train_examples = [
        core.Example(x=row[args.text_column], y=row[args.target_column])
        for _, row in train_data.iterrows()
    ]
    val_examples = [
        core.Example(x=row[args.text_column], y=row[args.target_column])
        for _, row in val_data.iterrows()
    ]

    # Initialize model
    model = model_lib.PyTorchModelConfig(
        architecture_kwargs=dict(num_encoder_layers=6, num_decoder_layers=6)
    ).make_model()

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        for i in range(0, len(train_examples), args.batch_size):
            batch = train_examples[i:i+args.batch_size]
            tensor_examples = model.converter.convert_examples(batch)

            optimizer.zero_grad()
            loss, _ = model.compute_losses_and_metrics(tensor_examples)
            loss.mean().backward()
            optimizer.step()

            if i % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs}, Batch {i//args.batch_size}, Loss: {loss.mean().item():.4f}")

        # Validation
        model.eval()
        val_losses = []
        for i in range(0, len(val_examples), args.batch_size):
            batch = val_examples[i:i+args.batch_size]
            tensor_examples = model.converter.convert_examples(batch)
            loss, _ = model.compute_losses_and_metrics(tensor_examples)
            val_losses.append(loss.mean().item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset (CSV).")
    parser.add_argument("--text_column", type=str, required=True, help="Name of the column containing text descriptions.")
    parser.add_argument("--target_column", type=str, required=True, help="Name of the column containing target values.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval.")
    args = parser.parse_args()
    main(args)
