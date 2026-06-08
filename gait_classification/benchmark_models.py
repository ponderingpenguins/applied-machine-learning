"""Benchmark saved gait-verification methods without retraining.

eg:

python -m gait_classification.benchmark_models \
    --output benchmark_results/benchmark_scores.json \
    --n-bootstrap 2000 \
    batch_size=128 embedding_size=32 dropout=0 transformer_d_model=32 \
    transformer_nhead=2 transformer_dim_feedforward=256 'preprocess_filters=[none]'
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

from gait_classification.data.filters import construct_filters
from gait_classification.data.gait_data import (
    apply_scaler,
    build_windowed_data,
    load_and_preprocess_data,
    participant_split,
)
from gait_classification.eval import bootstrap_far_frr_eer, compute_far_frr_eer
from gait_classification.models.cosface_head import CosFaceHead
from gait_classification.models.models import construct_model
from gait_classification.train import compute_embeddings
from gait_classification.utils import TrainConfig


def _parse_args() -> tuple[argparse.Namespace, TrainConfig]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transformer-checkpoint",
        default="checkpoints/final_model_transformer.pt",
    )
    parser.add_argument(
        "--lstm-checkpoint",
        default="checkpoints/best_model_lstm.pt",
    )
    parser.add_argument("--output", default="benchmark_results/benchmark_scores.json")
    parser.add_argument("--n-bootstrap", type=int, default=0)
    parser.add_argument("--random-runs", type=int, default=10)
    args, overrides = parser.parse_known_args()

    valid_fields = {field.name for field in fields(TrainConfig)}
    override_cfg = OmegaConf.from_dotlist(overrides)
    unknown = set(override_cfg.keys()) - valid_fields
    if unknown:
        parser.error(f"Unknown TrainConfig override(s): {', '.join(sorted(unknown))}")

    defaults = TrainConfig(
        model_type="transformer",
        loss_type="cosface",
        batch_size=128,
        embedding_size=32,
        dropout=0.0,
        transformer_d_model=32,
        transformer_nhead=2,
        transformer_num_layers=2,
        transformer_dim_feedforward=256,
        preprocess_filters=["none"],
    )
    cfg = OmegaConf.merge(OmegaConf.structured(defaults), override_cfg)
    return args, TrainConfig(**OmegaConf.to_container(cfg, resolve=True))


def _load_model(checkpoint_path: str, cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    for name in (
        "model_type",
        "embedding_size",
        "dropout",
        "lstm_hidden_size",
        "lstm_num_layers",
        "transformer_d_model",
        "transformer_nhead",
        "transformer_num_layers",
        "transformer_dim_feedforward",
    ):
        if name in checkpoint:
            setattr(cfg, name, checkpoint[name])

    base_model = construct_model(cfg, device)
    classifier_key = next(
        (key for key in state_dict if key.endswith("classifier.weight")),
        None,
    )
    if classifier_key is None:
        model = base_model
    else:
        num_classes = state_dict[classifier_key].shape[0]
        model = CosFaceHead(base_model, cfg.embedding_size, num_classes).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def _group_by_pid(features: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(pid): features[labels == pid] for pid in np.unique(labels)}


def _embeddings_for_checkpoint(
    checkpoint_path: str,
    model_type: str,
    cfg: TrainConfig,
    windows: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> dict[int, np.ndarray]:
    model_cfg = TrainConfig(**asdict(cfg))
    model_cfg.model_type = model_type
    model = _load_model(checkpoint_path, model_cfg, device)
    return compute_embeddings(model, windows, labels, device, model_cfg.batch_size)


def _fft_features(windows: np.ndarray) -> np.ndarray:
    fft = np.abs(np.fft.rfft(windows, axis=1))
    return fft.transpose(0, 2, 1).reshape(len(windows), -1).astype(np.float32)


def _select_fft_prefix(features: np.ndarray, threshold: float) -> np.ndarray:
    contributions = np.abs(features).sum(axis=0)
    contributions /= contributions.sum()
    n_keep = int(np.searchsorted(np.cumsum(contributions), threshold) + 1)
    return np.arange(n_keep)


def _metric_report(
    name: str,
    features_by_pid: dict[int, np.ndarray],
    cfg: TrainConfig,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    eer, far, frr = compute_far_frr_eer(
        features_by_pid,
        seed=cfg.seed,
        n_resamples=cfg.evaluation_resamples,
    )
    report: dict[str, object] = {
        "name": name,
        "eer": eer,
        "far": far,
        "frr": frr,
        "eer_percent": eer * 100,
        "far_percent": far * 100,
        "frr_percent": frr * 100,
    }
    if n_bootstrap:
        report["bootstrap_ci95"] = bootstrap_far_frr_eer(
            features_by_pid,
            seed=cfg.seed,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
    return report


def _summarize_random_runs(
    labels: np.ndarray,
    cfg: TrainConfig,
    n_runs: int,
) -> dict[str, object]:
    rng = np.random.default_rng(1000)
    runs = []
    for _ in range(n_runs):
        features = rng.normal(size=(len(labels), cfg.embedding_size)).astype(np.float32)
        eer, far, frr = compute_far_frr_eer(
            _group_by_pid(features, labels),
            seed=cfg.seed,
            n_resamples=cfg.evaluation_resamples,
        )
        runs.append((eer, far, frr))

    values = np.asarray(runs, dtype=float)
    report: dict[str, object] = {"name": "IID random embeddings", "n_runs": n_runs}
    for index, metric in enumerate(("eer", "far", "frr")):
        metric_values = values[:, index]
        std = float(metric_values.std(ddof=1)) if n_runs > 1 else 0.0
        sem = std / np.sqrt(n_runs)
        report[metric] = {
            "mean": float(metric_values.mean()),
            "std": std,
            "sem": float(sem),
            "mean_percent": float(metric_values.mean() * 100),
            "std_percent": float(std * 100),
            "sem_percent": float(sem * 100),
        }
    return report


def _print_table(results: list[dict[str, object]]) -> None:
    print("\nHoldout benchmark")
    print(f"{'Method':24s} {'EER':>20s} {'FAR':>8s} {'FRR':>8s}")
    for result in results:
        if "n_runs" in result:
            eer = result["eer"]
            print(
                f"{result['name']:24s} "
                f"{eer['mean_percent']:6.2f}% +/- {eer['sem_percent']:.2f} SEM "
                f"{result['far']['mean_percent']:7.2f}% "
                f"{result['frr']['mean_percent']:7.2f}%"
            )
            continue

        eer_text = f"{result['eer_percent']:.2f}%"
        ci = result.get("bootstrap_ci95")
        if ci:
            eer_ci = ci["eer"]
            eer_text = (
                f"{result['eer_percent']:.2f}% "
                f"[{eer_ci['ci95_lower_percent']:.2f}, "
                f"{eer_ci['ci95_upper_percent']:.2f}]"
            )
        print(
            f"{result['name']:24s} "
            f"{eer_text:>20s} "
            f"{result['far_percent']:7.2f}% "
            f"{result['frr_percent']:7.2f}%"
        )


def main() -> None:
    args, cfg = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    raw, labels = load_and_preprocess_data(cfg, construct_filters(cfg))
    windows, labels = build_windowed_data(cfg, raw, labels)

    train_pids, val_pids, test_pids = participant_split(np.unique(labels), cfg)
    development_pids = np.concatenate([train_pids, val_pids])
    development_mask = np.isin(labels, development_pids)
    test_mask = np.isin(labels, test_pids)

    scaler = StandardScaler().fit(windows[development_mask].reshape(-1, 6))
    scaled_windows = apply_scaler(windows, scaler)
    test_windows = scaled_windows[test_mask]
    test_labels = labels[test_mask]

    results = [
        _metric_report(
            "Transformer",
            _embeddings_for_checkpoint(
                args.transformer_checkpoint,
                "transformer",
                cfg,
                test_windows,
                test_labels,
                device,
            ),
            cfg,
            args.n_bootstrap,
            cfg.seed,
        ),
        _metric_report(
            "LSTM",
            _embeddings_for_checkpoint(
                args.lstm_checkpoint,
                "lstm",
                cfg,
                test_windows,
                test_labels,
                device,
            ),
            cfg,
            args.n_bootstrap,
            cfg.seed + 100,
        ),
    ]

    fft_all = _fft_features(windows)
    selected_fft = _select_fft_prefix(fft_all[development_mask], cfg.fft_threshold)
    fft_scaler = StandardScaler().fit(fft_all[development_mask][:, selected_fft])
    fft_test = fft_scaler.transform(fft_all[test_mask][:, selected_fft])
    results.append(
        _metric_report(
            "FFT + centroid",
            _group_by_pid(fft_test, test_labels),
            cfg,
            args.n_bootstrap,
            cfg.seed + 200,
        )
    )
    results.append(_summarize_random_runs(test_labels, cfg, args.random_runs))

    report = {
        "description": (
            "Saved-checkpoint holdout benchmark. No training is performed. "
            "Transformer, LSTM, FFT, and IID random controls use the same "
            "participant-disjoint holdout and enrollment/probe evaluator."
        ),
        "device": str(device),
        "config": asdict(cfg),
        "test_participants": test_pids.tolist(),
        "test_windows": int(len(test_labels)),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_table(results)
    print(f"\nSaved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
