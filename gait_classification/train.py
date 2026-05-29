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

import copy
import json
import logging
import pickle
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from gait_classification.cosface_loss import CosFaceLoss
from gait_classification.data.filters import construct_filters
from gait_classification.data.gait_data import (
    GaitWindowDataset,
    apply_scaler,
    build_windowed_data,
    fit_scaler,
    load_and_preprocess_data,
    make_kfold_splits,
    participant_split,
)
from gait_classification.eval import compute_far_frr_eer
from gait_classification.hf_utils import upload_model_from_training
from gait_classification.models.cosface_head import CosFaceHead
from gait_classification.models.models import construct_model
from gait_classification.triplet_loss import OnlineTripletLoss
from gait_classification.cosface_loss import CosFaceLoss
from gait_classification.utils import LossType, TrainConfig, ModelType, format_sectioned_summary

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
    model.train(mode=train)
    mode = "Train" if train else "Val"

    with torch.set_grad_enabled(train):
        pbar = tqdm(loader, desc=f"{mode} batches", leave=False, total=len(loader))
        for windows, labels in pbar:
            windows = windows.to(device)
            labels = labels.to(device)

            if loss_type == LossType.COSFACE:
                loss = criterion(model(windows), labels.long())
            else:
                embeddings = model.get_embeddings(windows) if isinstance(model, CosFaceHead) else model(windows)
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
            pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

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
    windows_tensor = torch.tensor(windows, dtype=torch.float32)
    loader = DataLoader(
        windows_tensor, batch_size=batch_size, shuffle=False, num_workers=0
    )

    if len(windows) == 0:
        return {}

    with torch.inference_mode():
        embeddings = torch.cat(
            [
                (model.get_embeddings(batch.to(device)) if isinstance(model, CosFaceHead) else model(batch.to(device)))
                for batch in loader
            ],
            dim=0,
        ).cpu().numpy()

    return {pid: embeddings[labels == pid] for pid in np.unique(labels)}


def summarize_fold_histories(
    fold_histories: list[dict[str, list[float]]]
) -> dict[str, object]:
    """Compute mean, std, and SEM curves across fold histories."""

    def _mean_std_sem(curves: list[np.ndarray], prefix: str) -> dict[str, list[float]]:
        min_len = min(len(curve) for curve in curves)
        stack = np.stack([curve[:min_len] for curve in curves], axis=0)
        std = stack.std(axis=0)
        sem = std / np.sqrt(stack.shape[0])
        return {
            f"{prefix}_mean": stack.mean(axis=0).tolist(),
            f"{prefix}_std": std.tolist(),
            f"{prefix}_sem": sem.tolist(),
        }

    train_curves = [np.asarray(history["train_loss"], dtype=float) for history in fold_histories]
    val_eer_curves = [
        np.asarray(history["val_eer"], dtype=float)
        for history in fold_histories
        if "val_eer" in history and len(history["val_eer"]) > 0
    ]

    summary: dict[str, object] = {"n_folds": len(fold_histories), **_mean_std_sem(train_curves, "train_loss")}

    if val_eer_curves:
        summary.update(_mean_std_sem(val_eer_curves, "val_eer"))

    best_val_eers = [
        float(history["best_val_eer"])
        for history in fold_histories
        if history.get("best_val_eer") is not None
    ]
    if best_val_eers:
        best_vals = np.asarray(best_val_eers, dtype=float)
        best_std = float(best_vals.std())
        best_sem = float(best_std / np.sqrt(len(best_vals)))
        summary.update(
            {
                "best_val_eer_mean": float(best_vals.mean()),
                "best_val_eer_std": best_std,
                "best_val_eer_sem": best_sem,
                "best_val_eer_values": best_vals.tolist(),
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

    def _subset(pids):
        if pids is None:
            return None, None
        mask = np.isin(labels, pids)
        return windows_scaled[mask], labels[mask]

    train_windows, train_labels = _subset(train_pids)
    val_windows, val_labels = _subset(val_pids)
    test_windows, test_labels = _subset(test_pids)

    loss_type = LossType(cfg.loss_type)
    base_model = construct_model(cfg, device)

    if loss_type == LossType.COSFACE:
        logger.info("Using CosFace loss for training.")
        unique_train_pids = sorted(np.unique(train_pids))
        pid_to_class_idx = {pid: i for i, pid in enumerate(unique_train_pids)}
        train_ds = GaitWindowDataset(train_windows, torch.tensor([pid_to_class_idx[pid] for pid in train_labels]))
        model = CosFaceHead(base_model, cfg.embedding_size, len(unique_train_pids)).to(device)
        train_criterion = CosFaceLoss(margin=cfg.cosface_margin, scale=cfg.cosface_scale)
    else:
        logger.info("Using Triplet loss for training.")
        train_ds = GaitWindowDataset(train_windows, train_labels)
        model = base_model
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    train_losses = []
    val_eers = []
    val_fars = []
    val_frrs = []
    best_val_eer = float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0
    early_stopping_patience = getattr(cfg, "early_stopping_patience", 0)
    early_stopping_min_delta = getattr(cfg, "early_stopping_min_delta", 0.0)

    logger.info("Starting training loop")
    for epoch in range(1, cfg.num_epochs + 1):
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

        if val_windows is not None:
            val_emb_by_pid = compute_embeddings(model, val_windows, val_labels, device, cfg.batch_size)
            val_eer, val_far, val_frr = compute_far_frr_eer(
                val_emb_by_pid,
                seed=cfg.seed,
                n_resamples=cfg.evaluation_resamples,
            )
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
                "Epoch %d/%d | train_loss=%.4f | val_eer=%.2f%% | val_far=%.2f%% | val_frr=%.2f%%",
                epoch,
                cfg.num_epochs,
                train_loss,
                val_eer * 100,
                val_far * 100,
                val_frr * 100,
            )

            if (
                early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                logger.info(
                    "Early stopping triggered after %d epochs without validation EER improvement.",
                    epochs_without_improvement,
                )
                break
        else:
            logger.info(
                "Epoch %d/%d  train_loss=%.4f",
                epoch,
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
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    history_fname = (
        f"training_history_fold{fold_idx}.json"
        if fold_idx is not None
        else "training_history.json"
    )
    history_path = os.path.join(cfg.checkpoint_dir, history_fname)
    history = {
        key: value
        for key, value in {
            "train_loss": train_losses,
            "val_eer": val_eers,
            "val_far": val_fars,
            "val_frr": val_frrs,
            "best_val_eer": best_val_eer if best_val_eer != float("inf") else None,
            "best_epoch": best_epoch if best_epoch >= 0 else None,
        }.items()
    }
    with open(history_path, "w") as f:
        json.dump(history, f)
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
        final_model_path = os.path.join(
            cfg.checkpoint_dir, f"final_model_{cfg.model_type}.pt"
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": cfg.model_type,
                "embedding_size": cfg.embedding_size,
                "input_size": 6,
                "lstm_hidden_size": cfg.lstm_hidden_size,
                "lstm_num_layers": cfg.lstm_num_layers,
                "dropout": cfg.dropout,
                "weight_decay": cfg.weight_decay,
                "transformer_d_model": cfg.transformer_d_model,
                "transformer_nhead": cfg.transformer_nhead,
                "transformer_num_layers": cfg.transformer_num_layers,
                "transformer_dim_feedforward": cfg.transformer_dim_feedforward,
            },
            final_model_path,
        )
        logger.info("Final model saved to %s", final_model_path)

    return history






def run_training(cfg: TrainConfig) -> None:
    """train the model"""
    logger.info("")
    logger.info("=== Training run ===")
    logger.info(
        "%s",
        format_sectioned_summary(
            "Configuration:",
            [
                (
                    "Optimization",
                    [
                        ("batch_size", cfg.batch_size),
                        ("num_epochs", cfg.num_epochs),
                        ("learning_rate", cfg.learning_rate),
                        ("weight_decay", cfg.weight_decay),
                        ("dropout", cfg.dropout),
                    ],
                ),
                (
                    "Metric/Loss",
                    [
                        ("model_type", cfg.model_type),
                        ("loss_type", cfg.loss_type),
                        ("embedding_size", cfg.embedding_size),
                        ("triplet_margin", cfg.triplet_margin),
                        ("cosface_margin", cfg.cosface_margin),
                        ("cosface_scale", cfg.cosface_scale),
                    ],
                ),
                (
                    "Architecture",
                    [
                        ("lstm_hidden_size", cfg.lstm_hidden_size),
                        ("lstm_num_layers", cfg.lstm_num_layers),
                        ("transformer_d_model", cfg.transformer_d_model),
                        ("transformer_nhead", cfg.transformer_nhead),
                        ("transformer_num_layers", cfg.transformer_num_layers),
                        ("transformer_dim_feedforward", cfg.transformer_dim_feedforward),
                    ],
                ),
                (
                    "Data",
                    [
                        ("n_folds", cfg.n_folds),
                        ("train_split", cfg.train_split),
                        ("val_split", cfg.val_split),
                        ("seq_len", cfg.seq_len),
                        ("window_stride", cfg.window_stride),
                        ("max_samples", cfg.max_samples),
                    ],
                ),
                (
                    "Checkpoints",
                    [("checkpoint_dir", cfg.checkpoint_dir)],
                ),
                (
                    "Data root",
                    [("data_dir", cfg.data_dir)],
                ),
            ],
        ),
    )
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    logger.info("")
    logger.info("=== Data loading ===")

    preprocess_functions = construct_filters(cfg)
    raw, y = load_and_preprocess_data(cfg, preprocess_functions=preprocess_functions)
    windows, labels = build_windowed_data(cfg, raw, y)
    logger.info("Loaded %d windows", len(windows))

    participants = np.unique(labels)
    logger.info("Found %d participants", len(participants))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    logger.info("Using device: %s", device)

    train_pids, val_pids, test_pids = participant_split(participants, cfg)
    development_pids = np.concatenate([train_pids, val_pids])

    fold_histories = []
    if cfg.n_folds and cfg.n_folds > 1:
        folds = make_kfold_splits(development_pids, cfg)
        for i, (t_pids, v_pids) in enumerate(folds):
            logger.info("")
            logger.info("=== Development fold %d/%d ===", i + 1, cfg.n_folds)
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
            if "val_eer_mean" in cv_summary:
                logger.info(
                    "CV summary: train_loss=%.4f | val_eer=%.2f%%",
                    cv_summary["train_loss_mean"][-1],
                    cv_summary["val_eer_mean"][-1] * 100,
                )
            else:
                logger.info(
                    "CV summary: train_loss=%.4f",
                    cv_summary["train_loss_mean"][-1],
                )

    logger.info("")
    logger.info("=== Final training on development participants ===")
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

    run_training(cfg)


if __name__ == "__main__":
    main()
