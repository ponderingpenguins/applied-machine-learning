import torch
import torch.nn as nn


def mine_semihard_triplets(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Mine semi-hard triplets per-label from a batch of embeddings.

    For each label class in the batch:
    - Generate all anchor-positive pairs within that class
    - For each pair, select a semi-hard negative from other classes where:
      d(a,p) < d(a,n) < d(a,p) + margin
    - Falls back to hardest negative if no semi-hard negative exists.
    """
    dist_mat = torch.cdist(embeddings, embeddings, p=2)
    unique_labels = torch.unique(labels)
    anchors, positives, negatives = [], [], []

    for label in unique_labels:
        pos_mask = labels == label
        neg_mask = ~pos_mask
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        if len(pos_indices) < 2 or len(neg_indices) == 0:
            continue

        pos_list = pos_indices.tolist()
        neg_list = neg_indices.tolist()

        for a_idx in pos_list:
            for p_idx in pos_list:
                if a_idx == p_idx:
                    continue
                ap_dist = dist_mat[a_idx, p_idx]

                neg_dists = dist_mat[a_idx, neg_indices]
                semi_hard = (neg_dists > ap_dist) & (neg_dists < ap_dist + margin)

                if semi_hard.any():
                    n_idx = neg_indices[semi_hard][torch.argmin(neg_dists[semi_hard])]
                else:
                    n_idx = neg_indices[torch.argmin(neg_dists)]

                anchors.append(a_idx)
                positives.append(p_idx)
                negatives.append(n_idx)

    if not anchors:
        return None

    return (
        torch.tensor(anchors, device=embeddings.device),
        torch.tensor(positives, device=embeddings.device),
        torch.tensor(negatives, device=embeddings.device),
    )


class OnlineTripletLoss(nn.Module):
    """Online triplet loss with semi-hard negative mining."""

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor | None, int]:
        """Compute triplet loss with online mining.

        Returns:
            (loss, num_triplets) where loss is None if no triplets found
        """
        triplet_indices = mine_semihard_triplets(embeddings, labels, self.margin)
        if triplet_indices is None:
            return None, 0

        a, p, n = triplet_indices
        loss = self.loss_fn(embeddings[a], embeddings[p], embeddings[n])
        return loss, len(a)
