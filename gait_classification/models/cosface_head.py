# gait_classification/models/cosface_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceHead(nn.Module):
    """
    A wrapper model that adds a CosFace-compatible linear layer (classifier)
    to a base embedding model.
    """

    def __init__(self, base_model: nn.Module, embedding_size: int, num_classes: int):
        super().__init__()
        self.base_model = base_model
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        # The final linear layer where weights are class prototypes
        self.classifier = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produces logits (cosine similarities) for CosFace loss.
        """
        # 1. Get embeddings from the base model
        embeddings = self.base_model(x)

        # 2. L2-normalize the embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        # 3. L2-normalize the classifier weights
        weights_norm = F.normalize(self.classifier.weight, p=2, dim=1)

        # 4. Calculate cosine similarities (logits)
        logits = F.linear(embeddings_norm, weights_norm)

        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        A helper method to get just the embeddings during evaluation.
        """
        return self.base_model(x)
