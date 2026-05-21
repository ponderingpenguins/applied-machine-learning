"""
Plot training curves from saved training history.

Usage:
python plot_training_curves.py --checkpoint-dir checkpoints --output figures/training_curves.png
"""

import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np


def _load_cv_history(checkpoint_dir: str) -> dict[str, np.ndarray] | None:
    cv_summary_path = os.path.join(checkpoint_dir, "training_history_cv_mean.json")
    fold_paths = sorted(glob(os.path.join(checkpoint_dir, "training_history_fold*.json")))

    if os.path.exists(cv_summary_path):
        with open(cv_summary_path, "r") as f:
            history = json.load(f)
        return {
            "train_mean": np.asarray(history["train_loss_mean"], dtype=float),
            "val_mean": np.asarray(history["val_loss_mean"], dtype=float),
            "train_std": np.asarray(history.get("train_loss_std", []), dtype=float),
            "val_std": np.asarray(history.get("val_loss_std", []), dtype=float),
        }

    if fold_paths:
        histories = []
        for fold_path in fold_paths:
            with open(fold_path, "r") as f:
                histories.append(json.load(f))

        train_curves = [np.asarray(history["train_loss"], dtype=float) for history in histories]
        val_curves = [np.asarray(history["val_loss"], dtype=float) for history in histories]
        min_train_len = min(len(curve) for curve in train_curves)
        min_val_len = min(len(curve) for curve in val_curves)
        train_stack = np.stack([curve[:min_train_len] for curve in train_curves], axis=0)
        val_stack = np.stack([curve[:min_val_len] for curve in val_curves], axis=0)
        return {
            "train_mean": train_stack.mean(axis=0),
            "val_mean": val_stack.mean(axis=0),
            "train_std": train_stack.std(axis=0),
            "val_std": val_stack.std(axis=0),
        }

    return None


def _load_final_history(checkpoint_dir: str) -> dict[str, np.ndarray] | None:
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    if not os.path.exists(history_path):
        return None

    with open(history_path, "r") as f:
        history = json.load(f)

    return {
        "train_loss": np.asarray(history.get("train_loss", []), dtype=float),
        "val_loss": np.asarray(history.get("val_loss", []), dtype=float),
    }


def plot_training_curves(
    checkpoint_dir: str = "checkpoints", output_path: str = None
) -> None:
    """Plot cross-validation mean curves and the final full-training curves.

    Args:
        checkpoint_dir: Directory containing training history files.
        output_path: Optional path to save the plot. If None, displays the plot.
    """
    cv_history = _load_cv_history(checkpoint_dir)
    final_history = _load_final_history(checkpoint_dir)

    if cv_history is None and final_history is None:
        raise FileNotFoundError(f"No training history found in {checkpoint_dir}")

    epochs = None
    if cv_history is not None:
        train_mean = cv_history["train_mean"]
        val_mean = cv_history["val_mean"]
        train_std = cv_history["train_std"]
        val_std = cv_history["val_std"]
        epochs = np.arange(1, len(train_mean) + 1)
    else:
        train_mean = final_history["train_loss"]
        val_mean = final_history["val_loss"]
        train_std = np.array([])
        val_std = np.array([])
        epochs = np.arange(1, len(train_mean) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        train_mean,
        "b-",
        label="Train Loss (mean)",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        epochs,
        val_mean,
        "r-",
        label="Val Loss (mean)",
        linewidth=2,
        marker="s",
        markersize=4,
    )

    if final_history is not None and len(final_history["train_loss"]) > 0 and cv_history is not None:
        final_epochs = np.arange(1, len(final_history["train_loss"]) + 1)
        plt.plot(
            final_epochs,
            final_history["train_loss"],
            color="navy",
            linestyle="--",
            label="Train Loss (final)",
            linewidth=1.8,
            alpha=0.9,
        )
    if final_history is not None and len(final_history["val_loss"]) > 0 and cv_history is not None:
        final_epochs = np.arange(1, len(final_history["val_loss"]) + 1)
        plt.plot(
            final_epochs,
            final_history["val_loss"],
            color="darkred",
            linestyle="--",
            label="Val Loss (final)",
            linewidth=1.8,
            alpha=0.9,
        )

    if train_std.size == len(train_mean):
        plt.fill_between(
            epochs,
            train_mean - train_std,
            train_mean + train_std,
            color="blue",
            alpha=0.12,
            linewidth=0,
        )
    if val_std.size == len(val_mean):
        plt.fill_between(
            epochs,
            val_mean - val_std,
            val_mean + val_std,
            color="red",
            alpha=0.12,
            linewidth=0,
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Cross-Validation Mean and Final Training Curves", fontsize=14, fontweight="bold")
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
        help="Directory containing training history files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the plot (e.g., training_curves.png). If not provided, displays the plot.",
    )

    args = parser.parse_args()
    plot_training_curves(args.checkpoint_dir, args.output)
