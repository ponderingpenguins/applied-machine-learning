import torch
import torch.nn as nn


class CosFaceLoss(nn.Module):
    """CosFace loss for face recognition and similar tasks."""

    def __init__(self, scale: float = 30.0, margin: float = 0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute CosFace loss.

        Args:
            logits: Cosine similarity scores (before scaling and margin)
            labels: Ground truth class labels

        Returns:
            Loss value
        """
        # Create one-hot encoding of labels
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply margin to the correct class logits
        adjusted_logits = logits - one_hot * self.margin

        # Scale the logits
        scaled_logits = adjusted_logits * self.scale

        # Compute cross-entropy loss
        loss = nn.CrossEntropyLoss()(scaled_logits, labels)
        return loss
