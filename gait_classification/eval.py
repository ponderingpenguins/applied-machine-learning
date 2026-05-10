import numpy as np


def compute_far_frr_eer(
    train_emb_by_pid: dict[int, np.ndarray],
    test_emb_by_pid: dict[int, np.ndarray],
    test_labels: np.ndarray,
    known_pids: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FAR, FRR, and EER using centroid-based distance.
    Mirrors the notebook's evaluation logic.
    """
    embedding_size = next(iter(train_emb_by_pid.values())).shape[1]
    centroids = np.zeros((len(known_pids), embedding_size), dtype=np.float32)

    for i, pid in enumerate(known_pids):
        if pid in train_emb_by_pid:
            centroids[i] = train_emb_by_pid[pid].mean(axis=0)
        else:
            centroids[i] = np.zeros(embedding_size)

    distances_known = []
    distances_unknown = []

    for pid in known_pids:
        if pid not in test_emb_by_pid:
            continue
        embeddings = test_emb_by_pid[pid]
        pid_idx = np.where(known_pids == pid)[0][0]
        centroid = centroids[pid_idx]

        dists = np.linalg.norm(embeddings - centroid, axis=1)
        distances_known.extend(dists.tolist())

    all_unknown_pids = [p for p in test_emb_by_pid.keys() if p not in known_pids]
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
