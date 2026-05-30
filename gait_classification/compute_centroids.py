"""Compute and save centroids from an existing checkpoint."""

import os
import pickle
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from gait_classification.data.gait_data import (
    build_windowed_data,
    load_and_preprocess_data,
)
from gait_classification.data.filters import construct_filters
from gait_classification.models.models import construct_model
from gait_classification.utils import TrainConfig


def main():
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainConfig(**cfg)
    except TypeError as e:
        print(f"Error: {e}\nUsage: python -m gait_classification.compute_centroids max_samples=500")
        sys.exit(1)

    checkpoint_path = os.path.join(
        os.path.dirname(__file__), cfg.checkpoint_dir, f"best_model_{cfg.model_type}.pt"
    )
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cpu")
    model = construct_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loading and preprocessing data...")
    preprocess_functions = construct_filters(cfg)
    raw, y = load_and_preprocess_data(cfg, preprocess_functions=preprocess_functions)
    windows, labels = build_windowed_data(cfg, raw, y)

    print(f"Computing embeddings for {len(windows)} windows...")
    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(windows), cfg.batch_size):
            batch = torch.tensor(windows[i : i + cfg.batch_size], dtype=torch.float32)
            batch = batch.to(device)
            emb = model(batch)
            embeddings_list.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)

    print("Computing centroids...")
    centroids = {}
    for pid in np.unique(labels):
        mask = labels == pid
        centroids[pid] = embeddings[mask].mean(axis=0)
        print(
            f"  Person {pid}: {(mask).sum()} windows, centroid norm={np.linalg.norm(centroids[pid]):.3f}"
        )

    centroids_path = os.path.join(
        os.path.dirname(__file__), cfg.checkpoint_dir, f"centroids_{cfg.model_type}.pkl"
    )
    with open(centroids_path, "wb") as f:
        pickle.dump(centroids, f)
    print(f"Saved {len(centroids)} centroids to {centroids_path}")


if __name__ == "__main__":
    main()
