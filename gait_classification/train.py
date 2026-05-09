"""
The training logic for gait classification (open set classification using triplet loss).

Data pipeline:
1. Load the dataset (gyro and accelerometer data)
2. Split the data participant wise into train/val/test sets, of size 70/15/15.
3. preprocess it (e.g., normalization, windowing).
4. Create triplets (anchor, positive, negative) for training.

Training loop:
5. For each epoch:
    a. For each batch of triplets:
        i. Forward pass through the model to get embeddings.
        ii. Compute the triplet loss.
        iii. Backpropagate and update model parameters.
6. Save the trained model.
- for later: Use k-fold cross-validation to evaluate the model's performance and robustness on the validation set, and select the best model based on validation performance.

Evaluation:
7. Evaluate the trained model on the test set.
    a. Compute FAR and FRR and plot the FAR-FRR curve and compute the EER (Equal Error Rate).
    b. (for later) Evaluate the model the same way as the paper "Deep Learning-Based Gait Recognition
Using Smartphones in the Wild" does, by evaluating on the latest 10% of the data for each participant, and computing the accuracy of the model on that data.
"""

import logging
import sys

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import TrainConfig

from gait_classification.data.gait_data import (
    GaitDataset,
    apply_scaler,
    build_windowed_data,
    fit_scaler,
    generate_triplets,
    load_and_preprocess_data,
    participant_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fooberino(cfg: TrainConfig) -> None:
    """train the model"""
    logger.info("Training with config: %s", cfg)

    logger.info("Loading and windowing data...")
    raw, y = load_and_preprocess_data(cfg, preprocess_functions=[])
    windows, labels = build_windowed_data(cfg, raw, y)
    logger.info("Total windows: %d", len(windows))

    participants = np.unique(labels)
    logger.info("Total participants: %d", len(participants))

    train_pids, val_pids, test_pids = participant_split(participants, cfg)
    logger.info(
        "Train: %d, Val: %d, Test: %d participants",
        len(train_pids),
        len(val_pids),
        len(test_pids),
    )

    # z-score normalization using only training data statistics
    scaler = fit_scaler(cfg, windows, labels, train_pids)
    windows = apply_scaler(windows, scaler)

    train_mask = np.isin(labels, train_pids)
    val_mask = np.isin(labels, val_pids)
    test_mask = np.isin(labels, test_pids)

    train_windows, train_labels = windows[train_mask], labels[train_mask]
    val_windows, val_labels = windows[val_mask], labels[val_mask]
    test_windows, test_labels = windows[test_mask], labels[test_mask]

    logger.info(
        "Train windows: %d, Val windows: %d, Test windows: %d",
        len(train_windows),
        len(val_windows),
        len(test_windows),
    )

    rng = np.random.default_rng(cfg.seed)

    logger.info("Generating triplets...")
    train_triplets = generate_triplets(
        train_labels, train_pids, n_neg_per_pair=5, rng=rng
    )
    val_triplets = generate_triplets(val_labels, val_pids, n_neg_per_pair=5, rng=rng)
    logger.info(
        "Train triplets: %d, Val triplets: %d", len(train_triplets), len(val_triplets)
    )

    train_ds = GaitDataset(train_windows, train_labels, train_triplets)
    val_ds = GaitDataset(val_windows, val_labels, val_triplets)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error("Error: %s\n\nUsage: python scratch.py", e)
        sys.exit(1)

    fooberino(cfg)


if __name__ == "__main__":
    main()
