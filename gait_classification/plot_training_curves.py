"""Plot cross-validation diagnostics for overfitting.

Usage:
python -m gait_classification.plot_training_curves \
    --checkpoint-dir checkpoints \
    --output figures/overfitting_diagnostics.png
"""

import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def _load_fold_histories(checkpoint_dir: str) -> list[dict[str, list[float]]]:
    fold_paths = sorted(glob(os.path.join(checkpoint_dir, "training_history_fold*.json")))
    if not fold_paths:
        raise FileNotFoundError(
            f"No training_history_fold*.json files found in {checkpoint_dir}"
        )

    histories = []
    for fold_path in fold_paths:
        with open(fold_path, "r", encoding="utf-8") as file:
            histories.append(json.load(file))
    return histories


def _mean_and_sem(
    histories: list[dict[str, list[float]]],
    metric: str,
    n_epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.empty(n_epochs, dtype=float)
    sem = np.empty(n_epochs, dtype=float)
    counts = np.empty(n_epochs, dtype=int)

    for epoch_index in range(n_epochs):
        values = np.asarray(
            [
                history[metric][epoch_index]
                for history in histories
                if len(history[metric]) > epoch_index
            ],
            dtype=float,
        )
        counts[epoch_index] = len(values)
        mean[epoch_index] = values.mean()
        sem[epoch_index] = (
            values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
        )
    return mean, sem, counts


def plot_training_curves(
    checkpoint_dir: str = "checkpoints",
    output_path: str | None = None,
) -> None:
    """Plot mean training loss and validation verification errors across folds.

    Later epochs include the folds that were still training, and the number of
    contributing folds is shown below the x-axis. Shaded regions show mean +/-
    one standard error across the contributing folds.

    The independent holdout set is deliberately not plotted because it should
    not be used for epoch selection.
    """
    histories = _load_fold_histories(checkpoint_dir)
    required_metrics = ("train_loss", "val_eer", "val_far", "val_frr")
    for metric in required_metrics:
        if not all(metric in history and history[metric] for history in histories):
            raise ValueError(f"Fold histories do not all contain non-empty {metric!r}")

    n_epochs = max(
        len(history[metric])
        for history in histories
        for metric in required_metrics
    )
    epochs = np.arange(1, n_epochs + 1)
    train_mean, train_sem, train_counts = _mean_and_sem(histories, "train_loss", n_epochs)
    verification_metrics = {
        "EER": _mean_and_sem(histories, "val_eer", n_epochs),
        "FAR": _mean_and_sem(histories, "val_far", n_epochs),
        "FRR": _mean_and_sem(histories, "val_frr", n_epochs),
    }

    figure, (loss_axis, error_axis) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    loss_axis.plot(
        epochs,
        train_mean,
        color="tab:blue",
        marker="o",
        linewidth=2,
        label="Mean training loss",
    )
    loss_axis.fill_between(
        epochs,
        train_mean - train_sem,
        train_mean + train_sem,
        color="tab:blue",
        alpha=0.18,
    )
    loss_axis.set_ylabel("CosFace training loss")
    loss_axis.set_title("Optimization on Training Participants")
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()

    colors = {"EER": "tab:red", "FAR": "tab:orange", "FRR": "tab:green"}
    for metric, (mean, sem, _) in verification_metrics.items():
        mean_percent = mean * 100
        sem_percent = sem * 100
        error_axis.plot(
            epochs,
            mean_percent,
            color=colors[metric],
            marker="o",
            linewidth=2,
            label=f"Validation {metric}",
        )
        error_axis.fill_between(
            epochs,
            mean_percent - sem_percent,
            mean_percent + sem_percent,
            color=colors[metric],
            alpha=0.12,
        )

    error_axis.legend(
        handles=[
            *error_axis.get_lines(),
            Patch(
                facecolor="gray",
                alpha=0.18,
                label="Shaded area: mean +/- SEM across active folds",
            ),
        ],
        ncol=2,
    )
    error_axis.set_xlabel("Epoch")
    error_axis.set_ylabel("Validation error rate (%)")
    error_axis.set_title("Generalization to Unseen Validation Participants")
    error_axis.grid(alpha=0.3)
    error_axis.set_xticks(epochs)
    error_axis.set_xticklabels(
        [f"{epoch}\n(n={count})" for epoch, count in zip(epochs, train_counts)]
    )

    figure.suptitle(
        f"Overfitting Diagnostics ({len(histories)}-Fold Participant-Level CV)",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        (
            "Lines are means over folds still active at each epoch; "
            "n below each epoch is the number of contributing folds."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        figure.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cross-validation diagnostics")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output")
    args = parser.parse_args()
    plot_training_curves(args.checkpoint_dir, args.output)
