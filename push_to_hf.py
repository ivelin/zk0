#!/usr/bin/env python3
"""Push zk0 SmolVLA model to Hugging Face Hub.

This script uploads a SmolVLA model checkpoint directory to Hugging Face Hub.
The checkpoint directory should contain config.json, pytorch_model.bin, and README.md files.

Usage:
    python push_to_hf.py /path/to/checkpoint/directory [--repo-id REPO_ID]

Arguments:
    checkpoint_dir: Path to directory containing model files (config.json, model.safetensors, README.md)

Options:
    --repo-id: Hugging Face repository ID (default: ivelin/zk0-smolvla-fl)

Environment Variables:
    HF_TOKEN: Hugging Face API token (loaded from .env file if available)

Examples:
    python push_to_hf.py outputs/2025-10-09_13-59-05/models/temp_model_round_30
    python push_to_hf.py /path/to/my/model/checkpoint/dir --repo-id myuser/my-model
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, skipping .env loading")

def main():
    parser = argparse.ArgumentParser(
        description="Push zk0 SmolVLA model checkpoint to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python push_to_hf.py outputs/2025-10-09_13-59-05/models/temp_model_round_30
    python push_to_hf.py /path/to/my/model/checkpoint/dir --repo-id myuser/my-model
        """
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to directory containing model files (config.json, model.safetensors, README.md)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ivelin/zk0-smolvla-fl",
        help="Hugging Face repository ID (default: ivelin/zk0-smolvla-fl)"
    )

    args = parser.parse_args()

    # Model directory from CLI argument
    model_dir = Path(args.checkpoint_dir)
    repo_id = args.repo_id

    # Validate checkpoint directory exists and contains required files
    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {model_dir}")

    required_files = ["config.json", "model.safetensors", "README.md"]
    missing_files = []
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        raise FileNotFoundError(f"Missing required files in {model_dir}: {missing_files}")

    print(f"✅ Checkpoint directory validated: {model_dir}")
    print(f"📁 Found files: {list(model_dir.glob('*'))}")

    # Get HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found. Please set it in your .env file or environment.")

    # Initialize API
    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        print(f"✅ Repo '{repo_id}' ready")
    except Exception as e:
        print(f"❌ Failed to create/verify repo: {e}")
        raise

    # Upload folder directly
    try:
        # Generate commit message based on checkpoint directory name
        checkpoint_name = model_dir.name
        commit_message = f"Upload zk0 SmolVLA federated learning checkpoint from {checkpoint_name}"

        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message
        )
        print(f"🚀 Successfully pushed model to https://huggingface.co/{repo_id}")
        print(f"📊 Uploaded {len(list(model_dir.glob('*')))} files from {checkpoint_name}")
    except Exception as e:
        print(f"❌ Failed to upload: {e}")
        raise

if __name__ == "__main__":
    main()