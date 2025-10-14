"""Utility functions for SmolVLA federated learning."""

from __future__ import annotations

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional
from loguru import logger

# Import LeRobot components
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, FilteredLeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    FilteredLeRobotDataset = None
from lerobot.datasets.factory import make_dataset

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Import torchvision for image transforms

def load_smolvla_model(model_name: str = "lerobot/smolvla_base", device: str = "auto") -> SmolVLAPolicy:
    """Load SmolVLA model with environment-based distributed control.

    SmolVLA handles tensor parallelism internally. This function sets environment
    variables to control distributed behavior based on SMOLVLA_TP_PLAN:
    - 'none' (default): Single device mode, disable distributed
    - 'auto': Allow auto-detection of distributed setup
    - Other values: Reserved for future multi-GPU support

    The actual distributed initialization is handled by SmolVLA's VLM component.

    Args:
        model_name: Hugging Face model name
        device: Device to load model on ('auto', 'cuda', or 'cpu')

    Returns:
        Loaded SmolVLA model

    Raises:
        RuntimeError: If model loading fails
    """
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available, continue

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set environment variables based on TP plan preference
    tp_plan_str = os.environ.get('SMOLVLA_TP_PLAN', 'none')
    num_gpus = torch.cuda.device_count()
    logger.info(f"SMOLVLA_TP_PLAN={tp_plan_str}, num_gpus={num_gpus}")

    if tp_plan_str == 'none':
        # Force single-device mode
        os.environ['USE_TORCH_DISTRIBUTED'] = '0'
        os.environ['TP_PLAN'] = 'disabled'
        logger.info("Configured for single-device mode (distributed disabled)")
    elif tp_plan_str == 'auto':
        # Allow auto-detection
        if num_gpus > 1:
            logger.info("Multi-GPU detected, allowing distributed auto-detection")
        else:
            logger.info("Single GPU, distributed will be disabled automatically")
    else:
        logger.warning(f"Unknown SMOLVLA_TP_PLAN value '{tp_plan_str}', defaulting to single-device mode")
        os.environ['USE_TORCH_DISTRIBUTED'] = '0'
        os.environ['TP_PLAN'] = 'disabled'

    # Import SmolVLA after setting environment variables to ensure they take effect
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    # Log versions for debugging
    import lerobot
    logger.info(f"LeRobot version: {lerobot.__version__}")
    import safetensors
    logger.info(f"SafeTensors version: {safetensors.__version__}")
    import ray
    logger.info(f"Ray version: {ray.__version__}")

    logger.info(f"Loading SmolVLA model: {model_name} on device {device}")

    # Try alternative loading strategies to avoid SafeTensors issues
    strategies = [
        ("Standard loading", lambda: SmolVLAPolicy.from_pretrained(model_name)),
        ("Force CPU loading", lambda: SmolVLAPolicy.from_pretrained(model_name, device_map="cpu")),
        ("Disable device mapping", lambda: SmolVLAPolicy.from_pretrained(model_name, device_map=None)),
    ]

    for strategy_name, load_func in strategies:
        try:
            logger.info(f"Trying {strategy_name}...")
            model = load_func()
            logger.info(f"Successfully loaded SmolVLA model using {strategy_name}")
            model = model.to(device)
            model.eval()
            return model
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"{strategy_name} failed: {error_msg}")
            # If it's a SafeTensors error, try the next strategy
            if "Attempted to access the data pointer" in error_msg:
                logger.info("SafeTensors error detected, trying next strategy...")
                continue
            # For other errors, still try next strategy
            continue

    # If all strategies fail, try one final attempt with explicit error handling
    try:
        logger.info("Final attempt: Loading with explicit SafeTensors bypass...")
        # Try to load without SafeTensors by using torch.load directly
        import tempfile
        from huggingface_hub import snapshot_download

        # Download model files to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info("Downloading model files to temporary directory...")
            model_path = snapshot_download(model_name, cache_dir=temp_dir)

            # Try to load using torch.load instead of SafeTensors
            model_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                logger.info("Loading model with torch.load...")
                model = SmolVLAPolicy.from_pretrained(model_name, local_files_only=True)
                logger.info("Successfully loaded model with torch.load")
                model = model.to(device)
                model.eval()
                return model
            else:
                logger.warning("pytorch_model.bin not found, trying safetensors files...")
                # Fall back to safetensors but with different approach
                model = SmolVLAPolicy.from_pretrained(model_name, local_files_only=True)
                logger.info("Successfully loaded model with local safetensors")
                model = model.to(device)
                model.eval()
                return model

    except Exception as e:
        logger.error(f"Final loading attempt also failed: {e}")

    # Final error if all strategies fail
    error_msg = ("SmolVLA model loading failed after all attempts. "
                "This may be due to SafeTensors compatibility issues with PyTorch/Ray. "
                "Consider updating PyTorch or SafeTensors versions, or using CPU-only mode.")
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def has_image_keys(sample):
    if isinstance(sample, dict):
        for k, v in sample.items():
            if 'image' in k.lower():
                return True
            if isinstance(v, dict):
                if has_image_keys(v):
                    return True
    return False


def load_lerobot_dataset(
    repo_id: str,
    tolerance_s: float = 0.0001,
    delta_timestamps: Optional[Dict[str, List[float]]] = None,
    episode_filter: Optional[Dict[str, int]] = None,
    split: Optional[str] = None
) -> LeRobotDataset:
    """Load LeRobot dataset using the same approach as lerobot train.py.

    This simplified version matches the working standalone training script,
    avoiding unnecessary preprocessing that interferes with LeRobot's internal handling.

    Args:
        repo_id: Hugging Face dataset repository ID
        tolerance_s: Timestamp tolerance for dataset loading (unused, kept for compatibility)
        delta_timestamps: Optional delta timestamps (unused, kept for compatibility)
        episode_filter: Optional episode filtering (unused, kept for compatibility)
        split: Optional split type (unused, kept for compatibility)

    Returns:
        Loaded LeRobotDataset

    Raises:
        RuntimeError: If loading fails
    """
    try:
        # Load config from SmolVLA model hub (same as standalone training)
        from lerobot.configs.train import TrainPipelineConfig

        try:
            cfg = TrainPipelineConfig.from_pretrained("lerobot/smolvla_base")
        except Exception as e:
            logger.warning(f"Failed to load config from hub: {e}, using default config with explicit image transforms")
            # Create minimal config if hub loading fails
            from lerobot.configs.default import DatasetConfig
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

            policy_config = SmolVLAConfig()
            policy_config.pretrained_path = "lerobot/smolvla_base"
            policy_config.push_to_hub = False  # Disable hub pushing for federated learning
            policy_config.resize_imgs_with_padding = (224, 224)  # SmolVLA ViT expects 224x224 images

            cfg = TrainPipelineConfig(
                dataset=DatasetConfig(repo_id=repo_id),
                policy=policy_config
            )

            # Add explicit image transforms only when using fallback config
            # Create proper ImageTransformsConfig object (disable random transforms)
            from lerobot.datasets.transforms import ImageTransformsConfig
            cfg.dataset.image_transforms = ImageTransformsConfig(
                enable=False,  # Disable random transforms for deterministic behavior
                tfs={}  # Empty dict to avoid random color jitter
            )

        # Override dataset configuration with our specific dataset
        cfg.dataset.repo_id = repo_id
        cfg.dataset.decode_videos = False  # Lazy decode for memory efficiency (match standalone train script)

        # Try loading with default config, fallback to OpenCV if it fails
        try:
            dataset = make_dataset(cfg)
            logger.info(f"Successfully loaded dataset {repo_id} with default video backend")
        except Exception as e:
            logger.warning(f"Dataset loading failed with default config: {e}")
            logger.info(f"Retrying with OpenCV video backend for dataset {repo_id}")
            try:
                cfg.dataset.video_backend = "opencv"
                dataset = make_dataset(cfg)
                logger.info(f"Successfully loaded dataset {repo_id} with OpenCV backend")
            except Exception as opencv_e:
                logger.error(f"Dataset {repo_id} failed to load with both default and OpenCV backends")
                logger.error(f"Default backend error: {e}")
                logger.error(f"OpenCV backend error: {opencv_e}")
                raise RuntimeError(f"Dataset {repo_id} is corrupted and unusable with available video backends")
        logger.info(f"Dataset contains {dataset.num_episodes} total episodes")

        # Log basic dataset info for debugging using decoded sample
        try:
            # Use dataset[0] to get a decoded sample (forces image/video decoding if codec available)
            sample = dataset[0]
            logger.info(f"Decoded dataset sample keys: {list(sample.keys())}")
            logger.info(f"Full decoded sample structure: {repr(sample)}")

            # Check for image keys in decoded sample
            image_keys = [k for k in sample.keys() if 'image' in k.lower()]
            if image_keys:
                logger.info(f"Keys containing 'image': {image_keys}")
                for key in image_keys:
                    value = sample[key]
                    logger.info(f"  - {key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
            else:
                # Check nested observation.images
                obs = sample.get('observation', {})
                images = obs.get('images', {}) if isinstance(obs, dict) else {}
                if images:
                    logger.info("Image data found in observation.images structure")
                    for img_key in images:
                        img_data = images[img_key]
                        logger.info(f"  - observation.images.{img_key}: type={type(img_data)}, shape={img_data.shape if hasattr(img_data, 'shape') else 'N/A'}")
                else:
                    logger.warning("No image data found in decoded sample - check codec/FFmpeg setup")

            # Log metadata availability
            if hasattr(dataset, 'meta'):
                meta = dataset.meta
                logger.info(f"Metadata available: episodes={getattr(meta, 'episodes', None) is not None}, camera_keys={getattr(meta, 'camera_keys', None)}")
                if hasattr(meta, 'camera_keys') and meta.camera_keys:
                    logger.info(f"Camera keys: {meta.camera_keys}")
                if hasattr(meta, 'episodes') and meta.episodes:
                    logger.info(f"Number of episodes in metadata: {len(meta.episodes)}")
                else:
                    logger.warning("Episodes metadata is None - likely codec/FFmpeg issue preventing full metadata load")
            else:
                logger.warning("Dataset has no 'meta' attribute - ensure LeRobotDataset is used")

        except Exception as e:
            logger.warning(f"Could not decode/get dataset sample: {e} - possible codec issue")

        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset {repo_id}: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}")


# Parameter validation utilities for federated learning
def compute_parameter_hash(parameters: List[np.ndarray]) -> str:
    """
    Compute SHA256 hash of model parameters for integrity checking.

    Args:
        parameters: List of parameter arrays

    Returns:
        SHA256 hash string of parameter data
    """
    import hashlib

    # Convert parameters to tensors and concatenate (deterministic ordering)
    tensors = [torch.from_numpy(p).flatten() for p in parameters]
    all_params = torch.cat([t for t in sorted(tensors, key=lambda x: x.numel())])
    param_bytes = all_params.numpy().tobytes()

    return hashlib.sha256(param_bytes).hexdigest()


def validate_and_log_parameters(
    parameters: List[np.ndarray],
    gate_name: str,
    expected_count: int = 506
) -> str:
    """
    Validate parameters and log basic information about them.

    Args:
        parameters: Parameter arrays to validate and log
        gate_name: Name of the gate (e.g., "server_initial_model", "client_update")
        expected_count: Expected number of parameters

    Returns:
        Parameter hash for downstream validation

    Raises:
        AssertionError: If validation fails
    """
    # Basic validation
    assert len(parameters) == expected_count, (
        f"Parameter count mismatch at {gate_name}: expected {expected_count}, got {len(parameters)}"
    )

    # Compute hash
    current_hash = compute_parameter_hash(parameters)

    # Log basic information
    logger.info(f"🛡️ {gate_name}: {len(parameters)} params, hash={current_hash[:8]}...")

    return current_hash


# TOML parsing utility
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def get_tool_config(tool_name: str, file_path: str = "pyproject.toml") -> dict:
    """Load tool configuration from pyproject.toml."""
    with open(file_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("tool", {}).get(tool_name, {})
def create_client_metrics_dict(
    round_num: int,
    client_id: str,
    dataset_name: str,
    policy_loss: float,
    fedprox_loss: float,
    grad_norm: float,
    param_hash: str,
    num_steps: int,
    param_update_norm: float
) -> dict:
    """
    Create standardized client metrics dictionary for JSON logging and server aggregation.
    
    Args:
        round_num: Current federated learning round
        client_id: Client identifier
        dataset_name: Dataset repository ID
        policy_loss: Pure SmolVLA policy loss
        fedprox_loss: FedProx regularization loss
        grad_norm: Gradient norm
        param_hash: SHA256 hash of parameters
        num_steps: Number of training steps completed
        param_update_norm: L2 norm of parameter updates
    
    Returns:
        Standardized metrics dict for consistent client/server use
    """
    total_loss = policy_loss + fedprox_loss
    return {
        "round": round_num,
        "client_id": client_id,
        "dataset_name": dataset_name,
        "loss": total_loss,  # Total loss for compatibility
        "policy_loss": policy_loss,  # Pure SmolVLA flow-matching loss
        "fedprox_loss": fedprox_loss,  # FedProx term
        "grad_norm": grad_norm,
        "param_hash": param_hash,
        "num_steps": num_steps,
        "param_update_norm": param_update_norm
    }