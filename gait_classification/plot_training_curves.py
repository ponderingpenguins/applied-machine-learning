"""
Plot training curves from saved training history.

Usage:
python plot_training_curves.py --checkpoint-dir checkpoints --output figures/training_curves.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    checkpoint_dir: str = "checkpoints", output_path: str = None
) -> None:
    """Plot training and validation loss curves.

    Args:
        checkpoint_dir: Directory containing training_history.json
        output_path: Optional path to save the plot. If None, displays the plot.
    """
    history_path = os.path.join(checkpoint_dir, "training_history.json")

    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history not found at {history_path}")

    with open(history_path, "r") as f:
        history = json.load(f)

    train_losses = history["train_loss"]
    val_losses = history["val_loss"]
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        train_losses,
        "b-",
        label="Train Loss",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        epochs,
        val_losses,
        "r-",
        label="Val Loss",
        linewidth=2,
        marker="s",
        markersize=4,
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss Curves", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory containing training_history.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the plot (e.g., training_curves.png). If not provided, displays the plot.",
    )

    args = parser.parse_args()
    plot_training_curves(args.checkpoint_dir, args.output)
