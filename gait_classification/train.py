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
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import ModelType, TrainConfig

from gait_classification.data.gait_data import (
    GaitDataset,
    apply_scaler,
    build_windowed_data,
    fit_scaler,
    generate_triplets,
    load_and_preprocess_data,
    participant_split,
)
from gait_classification.models.lstm import LSTM
from gait_classification.models.transformer import GaitTransformer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.TripletMarginLoss,
    optimizer=None,
    device: torch.device = None,
    train: bool = False,
) -> float:
    """Run a single epoch (training or validation)."""
    total_loss = 0.0
    n_batches = 0

    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = criterion(emb_a, emb_p, emb_n)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    cfg: TrainConfig,
) -> None:
    """Save a model checkpoint."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "model_type": cfg.model_type,
            "embedding_size": cfg.embedding_size,
            "input_size": 6,
            "hidden_size": 128,
            "num_layers": 2,
            "d_model": 64,
            "nhead": 4,
            "dim_feedforward": 256,
        },
        path,
    )
    logger.info("Checkpoint saved to %s (val_loss=%.4f)", path, val_loss)


def compute_embeddings(
    model: nn.Module,
    windows: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Compute embeddings for all windows, grouped by participant."""
    embeddings_list = []

    windows_tensor = torch.tensor(windows, dtype=torch.float32)
    loader = DataLoader(
        windows_tensor, batch_size=batch_size, shuffle=False, num_workers=0
    )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model(batch)
            embeddings_list.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)

    embeddings_by_pid = {}
    for pid in np.unique(labels):
        mask = labels == pid
        embeddings_by_pid[pid] = embeddings[mask]

    return embeddings_by_pid


def compute_far_frr_eer(
    train_emb_by_pid: dict[int, np.ndarray],
    test_emb_by_pid: dict[int, np.ndarray],
    test_labels: np.ndarray,
    known_pids: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FAR, FRR, and EER using centroid-based distance.
    Mirrors the notebook's evaluation logic.
    """
    embedding_size = next(iter(train_emb_by_pid.values())).shape[1]
    centroids = np.zeros((len(known_pids), embedding_size), dtype=np.float32)

    for i, pid in enumerate(known_pids):
        if pid in train_emb_by_pid:
            centroids[i] = train_emb_by_pid[pid].mean(axis=0)
        else:
            centroids[i] = np.zeros(embedding_size)

    distances_known = []
    distances_unknown = []

    for pid in known_pids:
        if pid not in test_emb_by_pid:
            continue
        embeddings = test_emb_by_pid[pid]
        pid_idx = np.where(known_pids == pid)[0][0]
        centroid = centroids[pid_idx]

        dists = np.linalg.norm(embeddings - centroid, axis=1)
        distances_known.extend(dists.tolist())

    all_unknown_pids = [p for p in test_emb_by_pid.keys() if p not in known_pids]
    for pid in all_unknown_pids:
        embeddings = test_emb_by_pid[pid]
        min_dists = np.min(
            np.linalg.norm(embeddings[:, None, :] - centroids[None, :, :], axis=2),
            axis=1,
        )
        distances_unknown.extend(min_dists.tolist())

    distances_known = np.array(distances_known)
    distances_unknown = np.array(distances_unknown)

    thresholds = np.linspace(
        0, np.max(np.concatenate([distances_known, distances_unknown])), 100
    )
    fars = []
    frrs = []

    for threshold in thresholds:
        far = (
            np.sum(distances_unknown < threshold) / len(distances_unknown)
            if len(distances_unknown) > 0
            else 0
        )
        frr = (
            np.sum(distances_known > threshold) / len(distances_known)
            if len(distances_known) > 0
            else 0
        )
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    eer_idx = np.argmin(np.abs(fars - frrs))
    eer = (fars[eer_idx] + frrs[eer_idx]) / 2

    return eer, thresholds, fars, frrs


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    logger.info("Using device: %s", device)

    if cfg.model_type == ModelType.LSTM or cfg.model_type == "lstm":
        logger.info("Using LSTM model")
        model = LSTM(
            input_size=6,
            hidden_size=128,
            num_layers=2,
            embedding_size=cfg.embedding_size,
        ).to(device)
    elif cfg.model_type == ModelType.TRANSFORMER or cfg.model_type == "transformer":
        logger.info("Using Transformer model")
        model = GaitTransformer(
            input_size=6,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            embedding_size=cfg.embedding_size,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")

    criterion = nn.TripletMarginLoss(margin=cfg.triplet_margin, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")

    logger.info("Starting training...")
    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )

        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(
                model, val_loader, criterion, None, device, train=False
            )

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch + 1,
            cfg.num_epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, val_loss, cfg)

    logger.info("Training complete. Evaluating on test set...")
    model.eval()

    train_emb_by_pid = compute_embeddings(
        model, train_windows, train_labels, device, cfg.batch_size
    )
    test_emb_by_pid = compute_embeddings(
        model, test_windows, test_labels, device, cfg.batch_size
    )

    eer, _, _, _ = compute_far_frr_eer(
        train_emb_by_pid, test_emb_by_pid, test_labels, train_pids
    )

    logger.info("Test EER: %.2f%%", eer * 100)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    final_model_path = os.path.join(cfg.checkpoint_dir, "final_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": cfg.model_type,
            "embedding_size": cfg.embedding_size,
            "input_size": 6,
            "hidden_size": 128,
            "num_layers": 2,
            "d_model": 64,
            "nhead": 4,
            "dim_feedforward": 256,
        },
        final_model_path,
    )
    logger.info("Final model saved to %s", final_model_path)


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
