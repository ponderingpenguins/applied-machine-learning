"""Iterative hyperparameter finetuning for gait classification.

This code follows the lecture suggestions:
- tune one parameter at a time;
- compare only the primary metric (for use that is validation EER);
- use SEM instead of std for the decision rule;
- keep the search spaces small and sensible.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
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
from gait_classification.triplet_loss import OnlineTripletLoss
from gait_classification.utils import ModelType, TrainConfig

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
		raise KeyError("Cross-validation summary did not include best_val_eer_mean/best_val_eer_sem")

	return float(summary["best_val_eer_mean"]), float(summary["best_val_eer_sem"])


def _compare_against_baseline(
	baseline_mean: float,
	baseline_sem: float,
	candidate_mean: float,
	candidate_sem: float,
) -> tuple[bool, float, float]:
	"""Return whether the candidate is a real improvement over the baseline."""

	delta = baseline_mean - candidate_mean
	combined_sem = float(np.sqrt(baseline_sem**2 + candidate_sem**2))
	threshold = 2.0 * combined_sem
	return delta > threshold, delta, threshold


def _plot_stage_results(
	stage_name: str,
	baseline_mean: float,
	baseline_sem: float,
	stage_results: list[dict[str, Any]],
	output_dir: str,
) -> str:
	os.makedirs(output_dir, exist_ok=True)

	labels = [result["label"] for result in stage_results]
	means = np.asarray([result["mean"] for result in stage_results], dtype=float)
	sems = np.asarray([result["sem"] for result in stage_results], dtype=float)

	x = np.arange(len(labels))
	plt.figure(figsize=(max(7, 1.4 * len(labels)), 4.8))
	plt.errorbar(
		x,
		means,
		yerr=sems,
		fmt="o",
		color="tab:blue",
		capsize=5,
		linewidth=2,
		label="candidate mean ± SEM",
	)
	plt.axhline(
		baseline_mean,
		color="tab:gray",
		linestyle="--",
		linewidth=2,
		label=f"baseline = {baseline_mean:.4f} ± {baseline_sem:.4f}",
	)
	plt.xticks(x, labels, rotation=20, ha="right")
	plt.ylabel("Validation EER")
	plt.xlabel(stage_name)
	plt.title(f"Finetuning stage: {stage_name}")
	plt.grid(True, axis="y", alpha=0.25)
	plt.legend()
	plt.tight_layout()

	output_path = os.path.join(output_dir, f"{stage_name}.png")
	plt.savefig(output_path, dpi=300)
	plt.close()
	return output_path


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
	criterion = OnlineTripletLoss(margin=cfg.triplet_margin)

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
				criterion=criterion,
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
	return ", ".join(f"{key}={value}" for key, value in updates.items())


def _tuning_plan(cfg: TrainConfig) -> list[tuple[str, list[CandidateUpdate]]]:
	shared_stages: list[tuple[str, list[CandidateUpdate]]] = [
		(
			"learning_rate",
			[
				{"learning_rate": 1e-4},
				{"learning_rate": 5e-4},
				{"learning_rate": 1e-3},
				{"learning_rate": 5e-3},
			],
		),
		(
			"triplet_margin",
			[
				{"triplet_margin": 0.2},
				{"triplet_margin": 0.3},
				{"triplet_margin": 0.4},
				{"triplet_margin": 0.5},
			],
		),
		(
			"embedding_size",
			[
				{"embedding_size": 32},
				{"embedding_size": 64},
				{"embedding_size": 128},
			],
		),
		(
			"weight_decay",
			[
				{"weight_decay": 1e-5},
				{"weight_decay": 1e-4},
				{"weight_decay": 5e-4},
			],
		),
		(
			"dropout",
			[
				{"dropout": 0.05},
				{"dropout": 0.1},
				{"dropout": 0.2},
			],
		),
	]

	if cfg.model_type == ModelType.LSTM or cfg.model_type == ModelType.LSTM.value:
		shared_stages.extend(
			[
				(
					"lstm_hidden_size",
					[
						{"lstm_hidden_size": 64},
						{"lstm_hidden_size": 128},
						{"lstm_hidden_size": 256},
					],
				),
				(
					"lstm_num_layers",
					[
						{"lstm_num_layers": 1},
						{"lstm_num_layers": 2},
						{"lstm_num_layers": 3},
					],
				),
			]
		)
	elif cfg.model_type == ModelType.TRANSFORMER or cfg.model_type == ModelType.TRANSFORMER.value:
		shared_stages.extend(
			[
				(
					"transformer_width",
					[
						{"transformer_d_model": 32, "transformer_nhead": 2},
						{"transformer_d_model": 64, "transformer_nhead": 4},
						{"transformer_d_model": 128, "transformer_nhead": 8},
					],
				),
				(
					"transformer_dim_feedforward",
					[
						{"transformer_dim_feedforward": 128},
						{"transformer_dim_feedforward": 256},
						{"transformer_dim_feedforward": 512},
					],
				),
				(
					"transformer_num_layers",
					[
						{"transformer_num_layers": 2},
						{"transformer_num_layers": 4},
						{"transformer_num_layers": 6},
					],
				),
			]
		)
	else:
		raise ValueError(f"Unknown model type for finetuning: {cfg.model_type}")

	return shared_stages


def _run_finetuning(cfg: TrainConfig) -> dict[str, Any]:
	logger.info("Finetuning with config: %s", cfg)

	preprocess_functions = construct_filters(cfg)
	raw, y = load_and_preprocess_data(cfg, preprocess_functions=preprocess_functions)
	windows, labels = build_windowed_data(cfg, raw, y)
	logger.info("Loaded %d windows for finetuning", len(windows))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("mps" if torch.backends.mps.is_available() else device)
	logger.info("Using device: %s", device)

	current_cfg = cfg
	current_eval = _evaluate_config(current_cfg, windows, labels, device)
	current_mean = current_eval["primary_mean"]
	current_sem = current_eval["primary_sem"]

	output_dir = os.path.join(cfg.checkpoint_dir, "finetuning")
	os.makedirs(output_dir, exist_ok=True)

	stage_reports: list[dict[str, Any]] = []
	for stage_name, candidates in _tuning_plan(cfg):
		baseline_mean = current_mean
		baseline_sem = current_sem
		logger.info(
			"Stage %s: baseline best_val_eer=%.4f ± %.4f",
			stage_name,
			baseline_mean,
			baseline_sem,
		)

		stage_results: list[dict[str, Any]] = []
		for candidate_updates in candidates:
			candidate_cfg = _clone_cfg(current_cfg, **candidate_updates)
			candidate_eval = _evaluate_config(candidate_cfg, windows, labels, device)
			candidate_mean = candidate_eval["primary_mean"]
			candidate_sem = candidate_eval["primary_sem"]
			accepted, delta, threshold = _compare_against_baseline(
				baseline_mean,
				baseline_sem,
				candidate_mean,
				candidate_sem,
			)

			stage_results.append(
				{
					"label": _candidate_label(candidate_updates),
					"updates": candidate_updates,
					"mean": candidate_mean,
					"sem": candidate_sem,
					"delta": delta,
					"threshold": threshold,
					"accepted": accepted,
				}
			)

			logger.info(
				"  %s -> best_val_eer=%.4f ± %.4f | delta=%.4f | 2xSEM=%.4f | %s",
				_candidate_label(candidate_updates),
				candidate_mean,
				candidate_sem,
				delta,
				threshold,
				"accept" if accepted else "reject",
			)

		accepted_candidates = [result for result in stage_results if result["accepted"]]
		if accepted_candidates:
			chosen = min(accepted_candidates, key=lambda result: result["mean"])
			current_cfg = _clone_cfg(current_cfg, **chosen["updates"])
			current_mean = chosen["mean"]
			current_sem = chosen["sem"]
			stage_decision = "accepted"
		else:
			chosen = None
			stage_decision = "kept baseline"

		plot_path = _plot_stage_results(
			stage_name,
			baseline_mean,
			baseline_sem,
			stage_results,
			output_dir,
		)
		stage_report = {
			"stage": stage_name,
			"baseline_mean": baseline_mean,
			"baseline_sem": baseline_sem,
			"results": stage_results,
			"decision": stage_decision,
			"chosen": chosen,
			"plot_path": plot_path,
		}
		stage_reports.append(stage_report)

		if chosen is not None:
			current_eval = {"primary_mean": current_mean, "primary_sem": current_sem}
		else:
			current_eval = {"primary_mean": baseline_mean, "primary_sem": baseline_sem}

	final_summary = {
		"model_type": cfg.model_type,
		"primary_metric": "best_val_eer",
		"final_best_val_eer_mean": current_mean,
		"final_best_val_eer_sem": current_sem,
		"best_config": current_cfg.__dict__,
		"stages": stage_reports,
	}

	summary_path = os.path.join(output_dir, "finetune_summary.json")
	with open(summary_path, "w") as f:
		json.dump(final_summary, f, indent=2)
	logger.info("Saved finetuning summary to %s", summary_path)
	logger.info(
		"Final selected config best_val_eer=%.4f ± %.4f",
		current_mean,
		current_sem,
	)

	return final_summary


def main() -> None:
	"""main function"""
	cfg = OmegaConf.structured(TrainConfig)
	cli_cfg = OmegaConf.from_cli()
	cfg = OmegaConf.merge(cfg, cli_cfg)
	cfg = OmegaConf.to_container(cfg, resolve=True)
	try:
		cfg = TrainConfig(**cfg)
	except TypeError as e:  # pylint: disable=broad-exception-raised
		logger.error("Error: %s\n\nUsage: python finetune.py", e)
		sys.exit(1)

	_run_finetuning(cfg)


if __name__ == "__main__":
	main()