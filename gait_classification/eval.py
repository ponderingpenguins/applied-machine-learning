import numpy as np


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

    distances_known = []
    distances_unknown = []

    for pid, embeddings in probe_known_embeddings:
        pid_idx = np.where(known_pids == pid)[0][0]
        centroid = centroids[pid_idx]
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        distances_known.extend(dists.tolist())

    for pid in unknown_pids:
        embeddings = test_emb_by_pid[pid]
        min_dists = np.min(
            np.linalg.norm(embeddings[:, None, :] - centroids[None, :, :], axis=2),
            axis=1,
        )
        distances_unknown.extend(min_dists.tolist())

    distances_known = np.asarray(distances_known, dtype=float)
    distances_unknown = np.asarray(distances_unknown, dtype=float)

    if len(distances_known) == 0 or len(distances_unknown) == 0:
        raise ValueError(
            "Open-set evaluation requires both known probe trials and unknown trials."
        )

    thresholds = np.linspace(
        0, np.max(np.concatenate([distances_known, distances_unknown])), 100
    )
    fars = []
    frrs = []

    for threshold in thresholds:
        far = np.sum(distances_unknown < threshold) / len(distances_unknown)
        frr = np.sum(distances_known > threshold) / len(distances_known)
        fars.append(far)
        frrs.append(frr)

    fars = np.asarray(fars, dtype=float)
    frrs = np.asarray(frrs, dtype=float)
    eer_idx = int(np.argmin(np.abs(fars - frrs)))

    eer = float((fars[eer_idx] + frrs[eer_idx]) / 2)
    far = float(fars[eer_idx])
    frr = float(frrs[eer_idx])
    return eer, far, frr


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
    rng = np.random.default_rng(seed)
    eligible_known_pids = [pid for pid, emb in test_emb_by_pid.items() if len(emb) >= 2]
    if not eligible_known_pids:
        raise ValueError(
            "Open-set evaluation requires at least one held-out participant with two or more windows."
        )

    shuffled_known = rng.permutation(eligible_known_pids)
    split_idx = max(1, len(shuffled_known) // 2)
    if split_idx == len(shuffled_known):
        split_idx = len(shuffled_known) - 1

    known_pids = np.array(sorted(shuffled_known[:split_idx]))
    unknown_pids = np.array(sorted(shuffled_known[split_idx:]))
    remaining_pids = np.array(sorted([pid for pid in test_emb_by_pid.keys() if pid not in eligible_known_pids]))
    unknown_pids = np.concatenate([unknown_pids, remaining_pids])
    if len(unknown_pids) == 0:
        raise ValueError(
            "Open-set evaluation requires both known probe participants and unknown participants."
        )

    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

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
