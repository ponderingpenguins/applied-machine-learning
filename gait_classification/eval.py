import numpy as np


def compute_far_frr_eer(
    test_emb_by_pid: dict[int, np.ndarray],
    seed: int = 67,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FAR, FRR, and EER using centroid-based distance.

    The known gallery is built from a deterministic enrollment split inside the
    held-out test participants. Distances for genuine trials are computed from
    the remaining probe windows of those same known participants.
    """
    rng = np.random.default_rng(seed)
    embedding_size = next(iter(test_emb_by_pid.values())).shape[1]
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
    if len(unknown_pids) == 0:
        raise ValueError(
            "Open-set evaluation requires both known probe participants and unknown participants."
        )

    centroids = np.zeros((len(known_pids), embedding_size), dtype=np.float32)
    probe_known_embeddings = []

    for i, pid in enumerate(known_pids):
        if pid not in test_emb_by_pid:
            continue

        embeddings = test_emb_by_pid[pid]
        if len(embeddings) == 0:
            continue

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

    all_unknown_pids = list(unknown_pids) + list(remaining_pids)
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
