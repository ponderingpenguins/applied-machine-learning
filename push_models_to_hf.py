#!/usr/bin/env python3
"""Script to push trained models to Hugging Face Hub.

Usage:
    python push_models_to_hf.py                  # Push all models from checkpoints/
    python push_models_to_hf.py --token YOUR_TOKEN  # Specify HF token explicitly
"""

import argparse
from pathlib import Path

from gait_classification.hf_utils import upload_models


def main():
    parser = argparse.ArgumentParser(
        description="Push trained models to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path(__file__).parent / "gait_classification" / "checkpoints",
        help="Directory containing model checkpoints (default: gait_classification/checkpoints)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (uses HF_TOKEN env var or cached credentials if not provided)",
    )
    args = parser.parse_args()

    if not args.checkpoints_dir.exists():
        print(f"Error: Checkpoints directory not found: {args.checkpoints_dir}")
        return 1

    print(f"Uploading models from {args.checkpoints_dir}...")
    upload_models(args.checkpoints_dir, token=args.token)
    print("All models uploaded successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
