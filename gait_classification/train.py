"""
The training logic for gait classification (open set classification using triplet loss).

Data pipeline:
1. Load the dataset (gyro and accelerometer data)
2. Split the data participant wise into train/val/test sets, of size 70/15/15.
3. preprocess it (e.g., normalization, windowing).
4. Create triplets (anchor, positive, negative) for training.

Training loop:
5. For each epoch:
    a. For each batch of triplets:
        i. Forward pass through the model to get embeddings.
        ii. Compute the triplet loss.
        iii. Backpropagate and update model parameters.
6. Save the trained model.
- for later: Use k-fold cross-validation to evaluate the model's performance and robustness on the validation set, and select the best model based on validation performance.

Evaluation:
7. Evaluate the trained model on the test set.
    a. Compute FAR and FRR and plot the FAR-FRR curve and compute the EER (Equal Error Rate).
    b. (for later) Evaluate the model the same way as the paper "Deep Learning-Based Gait Recognition
Using Smartphones in the Wild" does, by evaluating on the latest 10% of the data for each participant, and computing the accuracy of the model on that data.
"""

import logging
import sys

from omegaconf import OmegaConf
from utils import TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fooberino(cfg: TrainConfig) -> None:
    """train the model"""
    logger.info("Training with config: %s", cfg)


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error("Error: %s\n\nUsage: python scratch.py", e)
        sys.exit(1)

    fooberino(cfg)


if __name__ == "__main__":
    main()
