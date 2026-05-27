"""
The training logic for gait classification (open set classification using triplet loss).

Notes:
- online triplet mining based on: https://github.com/aktgpt/onlinetripletmining
- Todo: evaluate the model the same way as in the paper that we are comparing to

Transformer training example:
python train.py max_samples=500 batch_size=128 model_type=transformer 'preprocess_filters=[]'

LSTM training example:
python train.py max_samples=500 batch_size=128 model_type=lstm 'preprocess_filters=[]'
"""

import logging
import os
import sys
from typing import Optional

import numpy as np
from gait_classification.models.cosface_head import CosFaceHead
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from gait_classification.data.filters import construct_filters
from gait_classification.data.gait_data import (
    GaitWindowDataset,
    apply_scaler,
    build_windowed_data,
    fit_scaler,
    load_and_preprocess_data,
    participant_split,
    make_kfold_splits,
)
from gait_classification.eval import compute_far_frr_eer
from gait_classification.models.models import construct_model
from gait_classification.triplet_loss import OnlineTripletLoss
from gait_classification.cosface_loss import CosFaceLoss
from gait_classification.utils import LossType, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer=None,
    device: torch.device = None,
    train: bool = False,
    loss_type: str = LossType.TRIPLET
) -> float:
    """Run a single epoch with online semi-hard triplet mining."""
    total_loss = 0.0
    n_batches = 0
    total_batches = len(loader)

    mode = "Train" if train else "Val"
    pbar = tqdm(loader, desc=f"{mode} batches", leave=False, total=total_batches)
    for windows, labels in pbar:
        windows = windows.to(device)
        labels = labels.to(device)

        if loss_type == LossType.COSFACE:
            # For CosFace, the model returns logits and labels must be long
            logits = model(windows)
            loss = criterion(logits, labels)
        else: # Triplet Loss
            # For Triplet, we get embeddings. If it's a CosFaceHead model (in validation),
            # we need to call get_embeddings()
            if isinstance(model, CosFaceHead):
                embeddings = model.get_embeddings(windows)
            else:
                embeddings = model(windows)
            
            loss, n_triplets = criterion(embeddings, labels.long())
            if n_triplets == 0:
                continue

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        avg_loss = total_loss / n_batches
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    pbar.close()
    return total_loss / max(n_batches, 1)


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

    embeddings_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if isinstance(model, CosFaceHead):
                emb = model.get_embeddings(batch)
            else:
                emb = model(batch)
            embeddings_list.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)

    embeddings_by_pid = {}
    for pid in np.unique(labels):
        mask = labels == pid
        embeddings_by_pid[pid] = embeddings[mask]

    return embeddings_by_pid


def summarize_fold_histories(
    fold_histories: list[dict[str, list[float]]]
) -> dict[str, list[float]]:
    """Compute mean and std curves across fold histories."""
    train_curves = [np.asarray(history["train_loss"], dtype=float) for history in fold_histories]
    val_curves = [np.asarray(history["val_loss"], dtype=float) for history in fold_histories]

    min_train_len = min(len(curve) for curve in train_curves)
    min_val_len = min(len(curve) for curve in val_curves)

    train_stack = np.stack([curve[:min_train_len] for curve in train_curves], axis=0)
    val_stack = np.stack([curve[:min_val_len] for curve in val_curves], axis=0)

    return {
        "train_loss_mean": train_stack.mean(axis=0).tolist(),
        "train_loss_std": train_stack.std(axis=0).tolist(),
        "val_loss_mean": val_stack.mean(axis=0).tolist(),
        "val_loss_std": val_stack.std(axis=0).tolist(),
        "n_folds": len(fold_histories),
    }







def train_on_split(
    cfg: TrainConfig,
    windows: np.ndarray,
    labels: np.ndarray,
    train_pids,
    val_pids=None,
    test_pids=None,
    device: torch.device = None,
    fold_idx: Optional[int] = None,
    save_model: bool = False,
):
    """Train/evaluate on a single participant split, supporting multiple loss functions."""

    split_msg = "Train participants: %d, Val participants: %d" % (
        len(train_pids),
        len(val_pids) if val_pids is not None else 0,
    )
    if test_pids is not None:
        split_msg += ", Test participants: %d" % len(test_pids)
    logger.info(split_msg)

    # z-score normalization using only training data statistics
    scaler = fit_scaler(cfg, windows, labels, train_pids)
    windows_scaled = apply_scaler(windows, scaler)

    train_mask = np.isin(labels, train_pids)
    val_mask = np.isin(labels, val_pids) if val_pids is not None else None
    test_mask = np.isin(labels, test_pids) if test_pids is not None else None

    train_windows, train_labels = windows_scaled[train_mask], labels[train_mask]
    val_windows, val_labels = (
        (windows_scaled[val_mask], labels[val_mask]) if val_mask is not None else (None, None)
    )
    test_windows, test_labels = (
        (windows_scaled[test_mask], labels[test_mask]) if test_mask is not None else (None, None)
    )

    val_criterion = OnlineTripletLoss(margin=cfg.triplet_margin)
    
    if cfg.loss_type == LossType.COSFACE:
        logger.info("Using CosFace loss for training.")
        # Map training PIDs to 0-indexed classes
        unique_train_pids = sorted(np.unique(train_pids))
        pid_to_class_idx = {pid: i for i, pid in enumerate(unique_train_pids)}
        
        train_labels_mapped = torch.tensor([pid_to_class_idx[pid] for pid in train_labels])

        train_ds = GaitWindowDataset(train_windows, train_labels_mapped)
        # Note: We don't map val_labels, as they are for triplet loss validation
        val_ds = GaitWindowDataset(val_windows, val_labels) if val_windows is not None else None

        # Build CosFace model
        base_model = construct_model(cfg, device)
        model = CosFaceHead(base_model, cfg.embedding_size, len(unique_train_pids)).to(device)
        
        train_criterion = CosFaceLoss(margin=cfg.cosface_margin, scale=cfg.cosface_scale)

    else: # Default to Triplet Loss
        logger.info("Using Triplet loss for training.")
        train_ds = GaitWindowDataset(train_windows, train_labels)
        val_ds = GaitWindowDataset(val_windows, val_labels) if val_windows is not None else None
        
        model = construct_model(cfg, device)
        train_criterion = OnlineTripletLoss(margin=cfg.triplet_margin)

    train_labels = train_ds.labels.numpy()
    class_counts = np.bincount(train_labels)
    weights = 1.0 / class_counts[train_labels]
    train_sampler = WeightedRandomSampler(
        weights.tolist(), len(train_labels), replacement=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        if val_ds else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_losses = []
    val_losses = []

    logger.info("Starting training...")
    for epoch in tqdm(range(cfg.num_epochs), desc="Epochs"):
        model.train()
        train_loss = _run_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            device,
            train=True,
            loss_type=cfg.loss_type
        )

        train_losses.append(train_loss)

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = _run_epoch(
                    model,
                    val_loader,
                    val_criterion,
                    None,
                    device,
                    train=False,
                    loss_type=LossType.TRIPLET  # Always use triplet loss for validation
                )
            val_losses.append(val_loss)
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                epoch + 1,
                cfg.num_epochs,
                train_loss,
                val_loss,
            )
        else:
            logger.info(
                "Epoch %d/%d  train_loss=%.4f",
                epoch + 1,
                cfg.num_epochs,
                train_loss,
            )

    logger.info("Training complete. Saving history for this run...")
    import json

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    history_fname = (
        f"training_history_fold{fold_idx}.json" if fold_idx is not None else "training_history.json"
    )
    history_path = os.path.join(cfg.checkpoint_dir, history_fname)
    with open(history_path, "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f)
    logger.info("Training history saved to %s", history_path)

    if test_pids is not None and len(test_pids) > 0:
        logger.info("Evaluating on test set...")
        model.eval()

        test_emb_by_pid = compute_embeddings(
            model, test_windows, test_labels, device, cfg.batch_size
        )

        eer, _, _, _ = compute_far_frr_eer(
            test_emb_by_pid,
            seed=cfg.seed,
        )

        logger.info("Test EER: %.2f%%", eer * 100)

    if save_model:
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

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }






def fooberino(cfg: TrainConfig) -> None:
    """train the model"""
    logger.info("Training with config: %s", cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    logger.info("Loading and windowing data...")

    preprocess_functions = construct_filters(cfg)
    raw, y = load_and_preprocess_data(cfg, preprocess_functions=preprocess_functions)
    windows, labels = build_windowed_data(cfg, raw, y)
    logger.info("Total windows: %d", len(windows))

    participants = np.unique(labels)
    logger.info("Total participants: %d", len(participants))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    logger.info("Using device: %s", device)

    train_pids, val_pids, test_pids = participant_split(participants, cfg)
    development_pids = np.concatenate([train_pids, val_pids])

    fold_histories = []
    if cfg.n_folds and cfg.n_folds > 1:
        folds = make_kfold_splits(development_pids, cfg)
        for i, (t_pids, v_pids) in enumerate(folds):
            logger.info("=== Running development fold %d/%d ===", i + 1, cfg.n_folds)
            fold_history = train_on_split(
                cfg,
                windows,
                labels,
                t_pids,
                v_pids,
                test_pids=None,
                device=device,
                fold_idx=i + 1,
            )
            fold_histories.append(fold_history)

        if fold_histories:
            cv_summary = summarize_fold_histories(fold_histories)
            cv_summary_path = os.path.join(cfg.checkpoint_dir, "training_history_cv_mean.json")
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            import json

            with open(cv_summary_path, "w") as f:
                json.dump(cv_summary, f)
            logger.info(
                "Saved CV mean history to %s (folds=%d)",
                cv_summary_path,
                cv_summary["n_folds"],
            )
            logger.info(
                "CV mean final losses: train=%.4f val=%.4f",
                cv_summary["train_loss_mean"][-1],
                cv_summary["val_loss_mean"][-1],
            )

    logger.info("Retraining final model on all development participants before one test evaluation...")
    train_on_split(
        cfg,
        windows,
        labels,
        development_pids,
        val_pids=None,
        test_pids=test_pids,
        device=device,
        fold_idx=None,
        save_model=True,
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
