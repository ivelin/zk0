"""Push zk0 SmolVLA model checkpoint directory to Hugging Face Hub.

This script uploads a complete SmolVLA model checkpoint directory to Hugging Face Hub.
The checkpoint directory should contain all HF-required files (model.safetensors, config.json, README.md, etc.).

Usage:
    python -m zk0.push_to_hf /path/to/checkpoint_dir

Arguments:
    checkpoint_dir: Path to directory containing model files (model.safetensors, config.json, README.md, etc.)

Options:
    --repo-id: Hugging Face repository ID (default: ivelin/zk0-smolvla-fl)

Environment Variables:
    HF_TOKEN: Hugging Face API token (loaded from .env file if available)

Examples:
    python -m zk0.push_to_hf outputs/2025-10-09_13-59-05/models/checkpoint_round_30
    python -m zk0.push_to_hf /path/to/my/checkpoint/dir --repo-id myuser/my-model
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi
import json
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, skipping .env loading")


def push_to_hf(
    checkpoint_path: str, repo_id: str = "ivelin/zk0-smolvla-fl"
):
    """Push zk0 SmolVLA model checkpoint directory to Hugging Face Hub.

    This function uploads a complete SmolVLA model checkpoint directory to Hugging Face Hub.
    The directory should contain all HF-required files (model.safetensors, config.json, README.md, etc.).

    Args:
        checkpoint_path: Path to checkpoint directory containing all model files
        repo_id: Hugging Face repository ID (default: ivelin/zk0-smolvla-fl)

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist or missing required files
        ValueError: If HF_TOKEN is not set
        Exception: If upload fails
    """
    from src.server.server_utils import push_model_to_hub_enhanced
    from pathlib import Path
    import os

    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded from .env file")
    except ImportError:
        print("⚠️ python-dotenv not available, skipping .env loading")

    checkpoint_path = Path(checkpoint_path)

    # Validate it's a directory
    if not checkpoint_path.is_dir():
        raise ValueError(
            f"Invalid checkpoint path: {checkpoint_path}. Must be a checkpoint directory containing model files."
        )

    # Validate checkpoint directory exists and contains required files
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")

    required_files = ["model.safetensors", "config.json", "README.md"]
    missing_files = []
    for file in required_files:
        if not (checkpoint_path / file).exists():
            missing_files.append(file)

    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {checkpoint_path}: {missing_files}"
        )

    print(f"✅ Checkpoint directory validated: {checkpoint_path}")
    print(f"📁 Found files: {list(checkpoint_path.glob('*'))}")

    # Use the enhanced push function that uploads the entire directory
    push_model_to_hub_enhanced(checkpoint_path, repo_id)
    print(f"🚀 Successfully pushed model directory to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Push zk0 SmolVLA model checkpoint directory to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m zk0.push_to_hf outputs/2025-10-09_13-59-05/models/checkpoint_round_30
    python -m zk0.push_to_hf /path/to/my/checkpoint/dir --repo-id myuser/my-model
        """,
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint directory containing model files (model.safetensors, config.json, README.md, etc.)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ivelin/zk0-smolvla-fl",
        help="Hugging Face repository ID (default: ivelin/zk0-smolvla-fl)",
    )

    args = parser.parse_args()

    # Push the checkpoint directory
    push_to_hf(args.checkpoint_path, args.repo_id)


if __name__ == "__main__":
    main()
