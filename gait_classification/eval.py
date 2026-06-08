import numpy as np


def _split_known_unknown(
    embeddings_by_pid: dict[int, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(seed)
    eligible_known_pids = [
        pid for pid, embeddings in embeddings_by_pid.items() if len(embeddings) >= 2
    ]
    if not eligible_known_pids:
        raise ValueError(
            "Open-set evaluation requires at least one held-out participant with two or more windows."
        )

    shuffled_known = rng.permutation(eligible_known_pids)
    split_idx = max(1, len(shuffled_known) // 2)
    if split_idx == len(shuffled_known):
        split_idx -= 1

    known_pids = np.array(sorted(shuffled_known[:split_idx]))
    unknown_pids = np.array(sorted(shuffled_known[split_idx:]))
    remaining_pids = np.array(
        sorted([pid for pid in embeddings_by_pid if pid not in eligible_known_pids])
    )
    unknown_pids = np.concatenate([unknown_pids, remaining_pids])
    if len(unknown_pids) == 0:
        raise ValueError(
            "Open-set evaluation requires both known probe participants and unknown participants."
        )

    return known_pids, unknown_pids, rng


def _metrics_from_distances(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
) -> tuple[float, float, float]:
    if len(genuine_distances) == 0 or len(impostor_distances) == 0:
        raise ValueError("Open-set evaluation requires both genuine and impostors (among us).")

    thresholds = np.linspace(
        0,
        np.max(np.concatenate([genuine_distances, impostor_distances])),
        100,
    )
    fars = np.asarray(
        [np.sum(impostor_distances < threshold) / len(impostor_distances) for threshold in thresholds]
    )
    frrs = np.asarray(
        [np.sum(genuine_distances > threshold) / len(genuine_distances) for threshold in thresholds]
    )
    eer_idx = int(np.argmin(np.abs(fars - frrs)))
    return (
        float((fars[eer_idx] + frrs[eer_idx]) / 2),
        float(fars[eer_idx]),
        float(frrs[eer_idx]),
    )


def _compute_single_resample_metrics(
    test_emb_by_pid: dict[int, np.ndarray],
    known_pids: np.ndarray,
    unknown_pids: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    """Compute EER/FAR/FRR for one enrollment-probe resample."""
    rng = np.random.default_rng(seed)
    embedding_size = next(iter(test_emb_by_pid.values())).shape[1]

    centroids = np.zeros((len(known_pids), embedding_size), dtype=np.float32)
    probe_known_embeddings = []

    for i, pid in enumerate(known_pids):
        embeddings = test_emb_by_pid[pid]
        indices = rng.permutation(len(embeddings))
        enroll_count = max(1, len(indices) // 2)
        enroll_idx = indices[:enroll_count]
        probe_idx = indices[enroll_count:]

        centroids[i] = embeddings[enroll_idx].mean(axis=0)
        if len(probe_idx) > 0:
            probe_known_embeddings.append((pid, embeddings[probe_idx]))

    genuine_distances = []
    impostor_distances = []

    for pid, embeddings in probe_known_embeddings:
        pid_idx = np.where(known_pids == pid)[0][0]
        centroid = centroids[pid_idx]
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        genuine_distances.extend(dists.tolist())

    for pid in unknown_pids:
        embeddings = test_emb_by_pid[pid]
        min_dists = np.min(
            np.linalg.norm(embeddings[:, None, :] - centroids[None, :, :], axis=2),
            axis=1,
        )
        impostor_distances.extend(min_dists.tolist())

    return _metrics_from_distances(
        np.asarray(genuine_distances, dtype=float),
        np.asarray(impostor_distances, dtype=float),
    )


def compute_far_frr_eer(
    test_emb_by_pid: dict[int, np.ndarray],
    seed: int = 67,
    n_resamples: int = 10,
) -> tuple[float, float, float]:
    """
    Compute FAR, FRR, and EER using centroid-based distance.

    The known ket is built from a deterministic participant split inside the
    held-out set. The enrollment partition is then resampled multiple times
    and the reported metrics are averaged to reduce variance and any crazy instability.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    known_pids, unknown_pids, rng = _split_known_unknown(test_emb_by_pid, seed)
    resample_eers = []
    resample_fars = []
    resample_frrs = []

    for _ in range(n_resamples):
        resample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        eer, far, frr = _compute_single_resample_metrics(
            test_emb_by_pid,
            known_pids,
            unknown_pids,
            seed=resample_seed,
        )
        resample_eers.append(eer)
        resample_fars.append(far)
        resample_frrs.append(frr)

    eer_mean = float(np.mean(resample_eers))
    far_mean = float(np.mean(resample_fars))
    frr_mean = float(np.mean(resample_frrs))

    return eer_mean, far_mean, frr_mean


def bootstrap_far_frr_eer(
    embeddings_by_pid: dict[int, np.ndarray],
    seed: int = 67,
    n_bootstrap: int = 2000,
    bootstrap_seed: int | None = None,
) -> dict[str, dict[str, float]]:
    """Estimate participant/window bootstrap CIs for EER, FAR, and FRR.

    Known and unknown participants are resampled, and windows are resampled 
    within each selected participant. Each bootstrap rebuilds enrollment 
    centroids and computes the EER.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    known_pids, unknown_pids, _ = _split_known_unknown(embeddings_by_pid, seed)
    rng = np.random.default_rng(seed if bootstrap_seed is None else bootstrap_seed)
    samples = np.asarray(
        [
            _bootstrap_single_sample(embeddings_by_pid, known_pids, unknown_pids, rng)
            for _ in range(n_bootstrap)
        ],
        dtype=float,
    )

    intervals = {}
    for index, metric in enumerate(("eer", "far", "frr")):
        lower, upper = np.percentile(samples[:, index], [2.5, 97.5])
        intervals[metric] = {
            "mean": float(samples[:, index].mean()),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "mean_percent": float(samples[:, index].mean() * 100),
            "ci95_lower_percent": float(lower * 100),
            "ci95_upper_percent": float(upper * 100),
        }
    return intervals


def _bootstrap_single_sample(
    embeddings_by_pid: dict[int, np.ndarray],
    known_pids: np.ndarray,
    unknown_pids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    sampled_known = rng.choice(known_pids, size=len(known_pids), replace=True)
    sampled_unknown = rng.choice(unknown_pids, size=len(unknown_pids), replace=True)

    centroids = []
    genuine_distances = []
    for pid in sampled_known:
        embeddings = embeddings_by_pid[int(pid)]
        indices = rng.integers(0, len(embeddings), size=len(embeddings))
        enroll_count = max(1, len(indices) // 2)
        centroid = embeddings[indices[:enroll_count]].mean(axis=0)
        probes = embeddings[indices[enroll_count:]]
        centroids.append(centroid)
        if len(probes):
            genuine_distances.extend(np.linalg.norm(probes - centroid, axis=1))

    centroid_array = np.asarray(centroids)
    impostor_distances = []
    for pid in sampled_unknown:
        embeddings = embeddings_by_pid[int(pid)]
        indices = rng.integers(0, len(embeddings), size=len(embeddings))
        probes = embeddings[indices]
        distances = np.linalg.norm(
            probes[:, None, :] - centroid_array[None, :, :],
            axis=2,
        )
        impostor_distances.extend(np.min(distances, axis=1))

    return _metrics_from_distances(
        np.asarray(genuine_distances, dtype=float),
        np.asarray(impostor_distances, dtype=float),
    )
