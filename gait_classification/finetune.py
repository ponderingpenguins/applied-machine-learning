"""Grid search hyperparameter finetuning for gait classification."""

from __future__ import annotations

import itertools
import json
import logging
import os
import sys
import tempfile
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from gait_classification.data.filters import construct_filters
from gait_classification.data.gait_data import (
    build_windowed_data,
    load_and_preprocess_data,
    make_kfold_splits,
    participant_split,
)
from gait_classification.train import summarize_fold_histories, train_on_split
from gait_classification.utils import LossType, ModelType, TrainConfig, format_sectioned_summary

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CandidateUpdate = dict[str, Any]


def _clone_cfg(cfg: TrainConfig, **updates: Any) -> TrainConfig:
    return replace(cfg, **updates)


def _get_primary_metric(summary: dict[str, Any]) -> tuple[float, float]:
    if "best_val_eer_mean" not in summary or "best_val_eer_sem" not in summary:
        raise KeyError(
            "Cross-validation summary did not include best_val_eer_mean/best_val_eer_sem"
        )

    return float(summary["best_val_eer_mean"]), float(summary["best_val_eer_sem"])


def _combined_sem_threshold(best_sem: float, candidate_sem: float) -> float:
    return float(2.0 * np.sqrt(best_sem**2 + candidate_sem**2))


def _merge_candidate_updates(*updates: CandidateUpdate) -> CandidateUpdate:
    merged: CandidateUpdate = {}
    for update in updates:
        merged.update(update)
    return merged


def _evaluate_config(
    cfg: TrainConfig,
    windows: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    if cfg.n_folds < 2:
        raise ValueError("Finetuning requires at least 2 folds so SEM can be computed.")

    participants = np.unique(labels)
    train_pids, val_pids, _ = participant_split(participants, cfg)
    development_pids = np.concatenate([train_pids, val_pids])
    folds = make_kfold_splits(development_pids, cfg)

    fold_histories: list[dict[str, list[float]]] = []
    with tempfile.TemporaryDirectory(prefix="gait_finetune_") as temp_dir:
        trial_cfg = _clone_cfg(cfg, checkpoint_dir=temp_dir)
        for fold_idx, (train_fold_pids, val_fold_pids) in enumerate(folds, start=1):
            logger.info("Evaluating fold %d/%d for %s", fold_idx, cfg.n_folds, cfg.model_type)
            fold_history = train_on_split(
                trial_cfg,
                windows,
                labels,
                train_fold_pids,
                val_pids=val_fold_pids,
                test_pids=None,
                device=device,
                fold_idx=fold_idx,
                save_model=False,
            )
            fold_histories.append(fold_history)

    summary = summarize_fold_histories(fold_histories)
    primary_mean, primary_sem = _get_primary_metric(summary)
    return {
        "summary": summary,
        "primary_mean": primary_mean,
        "primary_sem": primary_sem,
    }


def _candidate_label(updates: CandidateUpdate) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(updates.items()))


def _candidate_complexity(updates: CandidateUpdate) -> float:
    """Return a rough proxy for model cost so simpler candidates sort first."""

    architecture_terms = {
        "embedding_size": 1.0,
        "lstm_hidden_size": 1.0,
        "lstm_num_layers": 32.0,
        "transformer_d_model": 1.0,
        "transformer_nhead": 4.0,
        "transformer_dim_feedforward": 0.25,
        "transformer_num_layers": 32.0,
    }
    return float(
        sum(
            float(updates[key]) * weight
            for key, weight in architecture_terms.items()
            if key in updates
        )
    )


def _build_grid(cfg: TrainConfig) -> list[CandidateUpdate]:
    """Build a flat list of candidate configurations to evaluate."""

    grid_dimensions: list[list[CandidateUpdate]] = [
        [{"learning_rate": value} for value in [1e-3]],
        [{"weight_decay": value} for value in [1e-5]],
        [{"dropout": value} for value in [0]],
        [{"embedding_size": value} for value in [16, 32]],
    ]

    loss_type = LossType(cfg.loss_type)
    if loss_type == LossType.TRIPLET:
        grid_dimensions.append([{"triplet_margin": value} for value in [0.2, 0.3, 0.4, 0.5]])
    elif loss_type == LossType.COSFACE:
        grid_dimensions.extend(
            [
                [{"cosface_margin": value} for value in [0.2, 0.3, 0.4]],
                [{"cosface_scale": value} for value in [16.0, 24.0, 30.0]],
            ]
        )
    else:
        raise ValueError(f"Unknown loss type for finetuning: {cfg.loss_type}")

    model_type = ModelType(cfg.model_type)
    if model_type == ModelType.LSTM:
        grid_dimensions.extend(
            [
                [{"lstm_hidden_size": value} for value in [64, 128]],
                [{"lstm_num_layers": value} for value in [1, 2, 3]],
            ]
        )
    elif model_type == ModelType.TRANSFORMER:
        grid_dimensions.extend(
            [
                [
                    {"transformer_d_model": 32, "transformer_nhead": 2},
                    {"transformer_d_model": 64, "transformer_nhead": 4},
                    {"transformer_d_model": 128, "transformer_nhead": 8},
                ],
                [{"transformer_dim_feedforward": value} for value in [128, 256]],
                [{"transformer_num_layers": value} for value in [2]],
            ]
        )
    else:
        raise ValueError(f"Unknown model type for finetuning: {cfg.model_type}")

    return [
        _merge_candidate_updates(*dimension_updates)
        for dimension_updates in itertools.product(*grid_dimensions)
    ]


def _evaluate_grid_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank the grid search outputs and identify the statistically tied top tier."""

    if not results:
        raise ValueError("Grid search produced no results.")

    sorted_results = sorted(
        results,
        key=lambda result: (
            result["mean"],
            result["sem"],
            _candidate_complexity(result["params"]),
            _candidate_label(result["params"]),
        ),
    )
    best = sorted_results[0]
    best_mean = float(best["mean"])
    best_sem = float(best["sem"])

    top_tier: list[dict[str, Any]] = []
    for result in sorted_results:
        threshold = _combined_sem_threshold(best_sem, float(result["sem"]))
        delta = float(result["mean"]) - best_mean
        result["delta_from_best"] = delta
        result["threshold"] = threshold
        result["within_best_threshold"] = delta <= threshold
        result["complexity"] = _candidate_complexity(result["params"])
        if result["within_best_threshold"]:
            top_tier.append(result)

    top_tier.sort(
        key=lambda result: (result["complexity"], result["mean"], result["sem"], result["label"])
    )

    return {
        "absolute_best": best,
        "top_tier": top_tier,
        "sorted_results": sorted_results,
    }


def _run_finetuning(cfg: TrainConfig) -> dict[str, Any]:
    logger.info("")
    logger.info("=== Grid search finetuning ===")
    logger.info(
        "%s",
        format_sectioned_summary(
            "Configuration:",
            [
                (
                    "Search",
                    [
                        ("model_type", cfg.model_type),
                        ("loss_type", cfg.loss_type),
                        ("n_folds", cfg.n_folds),
                        ("batch_size", cfg.batch_size),
                        ("num_epochs", cfg.num_epochs),
                    ],
                ),
                (
                    "Core params",
                    [
                        ("learning_rate", cfg.learning_rate),
                        ("weight_decay", cfg.weight_decay),
                        ("dropout", cfg.dropout),
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

    preprocess_functions = construct_filters(cfg)
    raw, y = load_and_preprocess_data(cfg, preprocess_functions=preprocess_functions)
    windows, labels = build_windowed_data(cfg, raw, y)
    logger.info("")
    logger.info("Loaded %d windows for grid search", len(windows))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    logger.info("Using device: %s", device)

    output_dir = os.path.join(cfg.checkpoint_dir, "finetuning")
    os.makedirs(output_dir, exist_ok=True)

    candidates = _build_grid(cfg)
    logger.info("")
    logger.info("Evaluating %d grid combinations", len(candidates))

    results: list[dict[str, Any]] = []
    for index, candidate_updates in enumerate(candidates, start=1):
        candidate_cfg = _clone_cfg(cfg, **candidate_updates)
        logger.info(
            "Candidate %d/%d | %s", index, len(candidates), _candidate_label(candidate_updates)
        )
        candidate_eval = _evaluate_config(candidate_cfg, windows, labels, device)
        candidate_mean = candidate_eval["primary_mean"]
        candidate_sem = candidate_eval["primary_sem"]
        results.append(
            {
                "params": candidate_updates,
                "label": _candidate_label(candidate_updates),
                "mean": candidate_mean,
                "sem": candidate_sem,
            }
        )
        logger.info("  best_val_eer=%.4f ± %.4f", candidate_mean, candidate_sem)

    analysis = _evaluate_grid_results(results)
    absolute_best = analysis["absolute_best"]
    top_tier = analysis["top_tier"]
    sorted_results = analysis["sorted_results"]

    logger.info("")
    logger.info("=== Grid search complete ===")
    logger.info(
        "Absolute best | best_val_eer=%.4f ± %.4f | %s",
        absolute_best["mean"],
        absolute_best["sem"],
        absolute_best["label"],
    )
    logger.info("Top-tier configurations: %d", len(top_tier))

    best_config = _clone_cfg(cfg, **absolute_best["params"])
    final_summary = {
        "model_type": cfg.model_type,
        "primary_metric": "best_val_eer",
        "grid_size": len(candidates),
        "best_result": {
            "label": absolute_best["label"],
            "params": absolute_best["params"],
            "mean": absolute_best["mean"],
            "sem": absolute_best["sem"],
        },
        "top_tier": [
            {
                "label": result["label"],
                "params": result["params"],
                "mean": result["mean"],
                "sem": result["sem"],
                "complexity": result["complexity"],
                "delta_from_best": result["delta_from_best"],
                "threshold": result["threshold"],
            }
            for result in top_tier
        ],
        "sorted_results": [
            {
                "label": result["label"],
                "params": result["params"],
                "mean": result["mean"],
                "sem": result["sem"],
                "complexity": result["complexity"],
                "delta_from_best": result["delta_from_best"],
                "threshold": result["threshold"],
                "within_best_threshold": result["within_best_threshold"],
            }
            for result in sorted_results
        ],
        "best_config": best_config.__dict__,
    }

    summary_path = os.path.join(output_dir, "grid_search_results.json")
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)
    logger.info("Saved grid search results to %s", summary_path)

    return final_summary


def main() -> None:
    """Main entry point."""
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    try:
        cfg = TrainConfig(**OmegaConf.to_container(cfg, resolve=True))
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error("Config error: %s\n\nUsage: python gait_classification/finetune.py", e)
        sys.exit(1)

    _run_finetuning(cfg)


if __name__ == "__main__":
    main()
