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
import copy
import sys
from typing import Optional

import numpy as np
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
from gait_classification.utils import TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: OnlineTripletLoss,
    optimizer=None,
    device: torch.device = None,
    train: bool = False,
) -> float:
    """Run a single epoch with online semi-hard triplet mining."""
    total_loss = 0.0
    n_batches = 0
    total_batches = len(loader)

    mode = "Train" if train else "Val"
    pbar = tqdm(loader, desc=f"{mode} batches", leave=False, total=total_batches)
    for windows, labels in pbar:
        windows = windows.to(device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        embeddings = model(windows)
        loss, n_triplets = criterion(embeddings, labels)

        if loss is None:
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


def summarize_fold_histories(
    fold_histories: list[dict[str, list[float]]]
) -> dict[str, list[float]]:
    """Compute mean and std curves across fold histories."""
    train_curves = [np.asarray(history["train_loss"], dtype=float) for history in fold_histories]
    val_loss_curves = [np.asarray(history["val_loss"], dtype=float) for history in fold_histories]
    val_eer_curves = [np.asarray(history["val_eer"], dtype=float) for history in fold_histories if "val_eer" in history]

    min_train_len = min(len(curve) for curve in train_curves)
    min_val_loss_len = min(len(curve) for curve in val_loss_curves)

    train_stack = np.stack([curve[:min_train_len] for curve in train_curves], axis=0)
    val_loss_stack = np.stack([curve[:min_val_loss_len] for curve in val_loss_curves], axis=0)

    summary = {
        "train_loss_mean": train_stack.mean(axis=0).tolist(),
        "train_loss_std": train_stack.std(axis=0).tolist(),
        "val_loss_mean": val_loss_stack.mean(axis=0).tolist(),
        "val_loss_std": val_loss_stack.std(axis=0).tolist(),
        "n_folds": len(fold_histories),
    }

    if val_eer_curves:
        min_val_eer_len = min(len(curve) for curve in val_eer_curves)
        val_eer_stack = np.stack([curve[:min_val_eer_len] for curve in val_eer_curves], axis=0)
        summary.update(
            {
                "val_eer_mean": val_eer_stack.mean(axis=0).tolist(),
                "val_eer_std": val_eer_stack.std(axis=0).tolist(),
            }
        )

    return summary







def train_on_split(
    cfg: TrainConfig,
    windows: np.ndarray,
    labels: np.ndarray,
    train_pids,
    val_pids=None,
    test_pids=None,
    device: torch.device = None,
    criterion: OnlineTripletLoss = None,
    fold_idx: Optional[int] = None,
    save_model: bool = False,
):
    """Train/evaluate on a single participant split."""

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

    logger.info(
        "Train windows: %d, Val windows: %d%s",
        len(train_windows),
        len(val_windows) if val_windows is not None else 0,
        ", Test windows: %d" % len(test_windows) if test_windows is not None else "",
    )

    train_ds = GaitWindowDataset(train_windows, train_labels)
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
            GaitWindowDataset(val_windows, val_labels),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        if val_windows is not None
        else None
    )

    model = construct_model(cfg, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_losses = []
    val_losses = []
    val_eers = []
    val_fars = []
    val_frrs = []
    best_val_eer = float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0
    early_stopping_patience = getattr(cfg, "early_stopping_patience", 0)
    early_stopping_min_delta = getattr(cfg, "early_stopping_min_delta", 0.0)

    logger.info("Starting training...")
    for epoch in tqdm(range(cfg.num_epochs), desc="Epochs"):
        model.train()
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
        )

        train_losses.append(train_loss)

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = _run_epoch(
                    model,
                    val_loader,
                    criterion,
                    None,
                    device,
                    train=False,
                )
                val_emb_by_pid = compute_embeddings(
                    model, val_windows, val_labels, device, cfg.batch_size
                )
                val_eer, val_far, val_frr = compute_far_frr_eer(
                    val_emb_by_pid,
                    seed=cfg.seed,
                    n_resamples=cfg.evaluation_resamples,
                )
            val_losses.append(val_loss)
            val_eers.append(val_eer)
            val_fars.append(val_far)
            val_frrs.append(val_frr)

            if val_eer + early_stopping_min_delta < best_val_eer:
                best_val_eer = val_eer
                best_epoch = epoch + 1
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_eer=%.2f%%  val_far=%.2f%%  val_frr=%.2f%%",
                epoch + 1,
                cfg.num_epochs,
                train_loss,
                val_loss,
                val_eer * 100,
                val_far * 100,
                val_frr * 100,
            )

            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %d epochs without validation EER improvement.",
                    epochs_without_improvement,
                )
                break
        else:
            logger.info(
                "Epoch %d/%d  train_loss=%.4f",
                epoch + 1,
                cfg.num_epochs,
                train_loss,
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(
            "Restored best model from epoch %d with validation EER %.2f%%",
            best_epoch,
            best_val_eer * 100,
        )

    logger.info("Training complete. Saving history for this run...")
    import json

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    history_fname = (
        f"training_history_fold{fold_idx}.json" if fold_idx is not None else "training_history.json"
    )
    history_path = os.path.join(cfg.checkpoint_dir, history_fname)
    with open(history_path, "w") as f:
        json.dump(
            {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "val_eer": val_eers,
                "val_far": val_fars,
                "val_frr": val_frrs,
                "best_val_eer": best_val_eer if best_val_eer != float("inf") else None,
                "best_epoch": best_epoch if best_epoch >= 0 else None,
            },
            f,
        )
    logger.info("Training history saved to %s", history_path)

    if test_pids is not None and len(test_pids) > 0:
        logger.info("Evaluating on test set...")
        model.eval()

        test_emb_by_pid = compute_embeddings(
            model, test_windows, test_labels, device, cfg.batch_size
        )

        eer, _, _ = compute_far_frr_eer(
            test_emb_by_pid,
            seed=cfg.seed,
            n_resamples=cfg.evaluation_resamples,
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
        "val_eer": val_eers,
        "val_far": val_fars,
        "val_frr": val_frrs,
        "best_val_eer": best_val_eer if best_val_eer != float("inf") else None,
        "best_epoch": best_epoch if best_epoch >= 0 else None,
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

    criterion = OnlineTripletLoss(margin=cfg.triplet_margin)

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
                criterion=criterion,
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
            if "val_eer_mean" in cv_summary:
                logger.info(
                    "CV mean final metrics: train_loss=%.4f val_loss=%.4f val_eer=%.2f%%",
                    cv_summary["train_loss_mean"][-1],
                    cv_summary["val_loss_mean"][-1],
                    cv_summary["val_eer_mean"][-1] * 100,
                )
            else:
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
        criterion=criterion,
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
