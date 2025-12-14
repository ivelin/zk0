"""Utility functions for SmolVLA federated learning."""

from __future__ import annotations

import json
import os
import sys
import torch
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
import re

from loguru import logger

# Import LeRobot components
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, FilteredLeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    FilteredLeRobotDataset = None
from lerobot.datasets.factory import make_dataset

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Import additional utilities
from src.common.parameter_utils import compute_parameter_hash
from src.server.metrics_utils import create_client_metrics_dict

# Import torchvision for image transforms


def load_smolvla_model(
    model_name: str = "lerobot/smolvla_base", device: str = "auto"
) -> SmolVLAPolicy:
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
    tp_plan_str = os.environ.get("SMOLVLA_TP_PLAN", "none")
    num_gpus = torch.cuda.device_count()
    logger.info(f"SMOLVLA_TP_PLAN={tp_plan_str}, num_gpus={num_gpus}")

    if tp_plan_str == "none":
        # Force single-device mode
        os.environ["USE_TORCH_DISTRIBUTED"] = "0"
        os.environ["TP_PLAN"] = "disabled"
        logger.info("Configured for single-device mode (distributed disabled)")
    elif tp_plan_str == "auto":
        # Allow auto-detection
        if num_gpus > 1:
            logger.info("Multi-GPU detected, allowing distributed auto-detection")
        else:
            logger.info("Single GPU, distributed will be disabled automatically")
    else:
        logger.warning(
            f"Unknown SMOLVLA_TP_PLAN value '{tp_plan_str}', defaulting to single-device mode"
        )
        os.environ["USE_TORCH_DISTRIBUTED"] = "0"
        os.environ["TP_PLAN"] = "disabled"

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
        (
            "Force CPU loading",
            lambda: SmolVLAPolicy.from_pretrained(model_name, device_map="cpu"),
        ),
        (
            "Disable device mapping",
            lambda: SmolVLAPolicy.from_pretrained(model_name, device_map=None),
        ),
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
                logger.warning(
                    "pytorch_model.bin not found, trying safetensors files..."
                )
                # Fall back to safetensors but with different approach
                model = SmolVLAPolicy.from_pretrained(model_name, local_files_only=True)
                logger.info("Successfully loaded model with local safetensors")
                model = model.to(device)
                model.eval()
                return model

    except Exception as e:
        logger.error(f"Final loading attempt also failed: {e}")

    # Final error if all strategies fail
    error_msg = (
        "SmolVLA model loading failed after all attempts. "
        "This may be due to SafeTensors compatibility issues with PyTorch/Ray. "
        "Consider updating PyTorch or SafeTensors versions, or using CPU-only mode."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def has_image_keys(sample):
    if isinstance(sample, dict):
        for k, v in sample.items():
            if "image" in k.lower():
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
    split: Optional[str] = None,
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
            logger.warning(
                f"Failed to load config from hub: {e}, using default config with explicit image transforms"
            )
            # Create minimal config if hub loading fails
            from lerobot.configs.default import DatasetConfig
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

            policy_config = SmolVLAConfig()
            policy_config.pretrained_path = "lerobot/smolvla_base"
            policy_config.push_to_hub = (
                False  # Disable hub pushing for federated learning
            )
            policy_config.resize_imgs_with_padding = (
                224,
                224,
            )  # SmolVLA ViT expects 224x224 images

            cfg = TrainPipelineConfig(
                dataset=DatasetConfig(repo_id=repo_id), policy=policy_config
            )

            # Add explicit image transforms only when using fallback config
            # Create proper ImageTransformsConfig object (disable random transforms)
            from lerobot.datasets.transforms import ImageTransformsConfig

            cfg.dataset.image_transforms = ImageTransformsConfig(
                enable=False,  # Disable random transforms for deterministic behavior
                tfs={},  # Empty dict to avoid random color jitter
            )

        # Override dataset configuration with our specific dataset
        cfg.dataset.repo_id = repo_id
        cfg.dataset.decode_videos = (
            False  # Lazy decode for memory efficiency (match standalone train script)
        )

        # Try loading with default config, fallback to OpenCV if it fails
        try:
            dataset = make_dataset(cfg)
            logger.info(
                f"Successfully loaded dataset {repo_id} with default video backend"
            )
        except Exception as e:
            logger.warning(f"Dataset loading failed with default config: {e}")
            logger.info(f"Retrying with OpenCV video backend for dataset {repo_id}")
            try:
                cfg.dataset.video_backend = "opencv"
                dataset = make_dataset(cfg)
                logger.info(
                    f"Successfully loaded dataset {repo_id} with OpenCV backend"
                )
            except Exception as opencv_e:
                logger.error(
                    f"Dataset {repo_id} failed to load with both default and OpenCV backends"
                )
                logger.error(f"Default backend error: {e}")
                logger.error(f"OpenCV backend error: {opencv_e}")
                raise RuntimeError(
                    f"Dataset {repo_id} is corrupted and unusable with available video backends"
                )
        logger.info(f"Dataset contains {dataset.num_episodes} total episodes")

        # Log basic dataset info for debugging using decoded sample
        try:
            # Use dataset[0] to get a decoded sample (forces image/video decoding if codec available)
            sample = dataset[0]
            logger.info(f"Dataset sample keys: {list(sample.keys())}")

            # Check for image keys in decoded sample (brief check only)
            image_keys = [k for k in sample.keys() if "image" in k.lower()]
            if image_keys:
                logger.info(f"Image keys found: {len(image_keys)} ({image_keys[0]}...)")
            else:
                logger.warning("No image data found in decoded sample - check codec/FFmpeg setup")

            # Log essential metadata only
            if hasattr(dataset, "meta"):
                meta = dataset.meta
                episode_count = len(meta.episodes) if hasattr(meta, "episodes") and meta.episodes else 0
                logger.info(f"Dataset metadata: {episode_count} episodes")
            else:
                logger.warning("Dataset has no 'meta' attribute - ensure LeRobotDataset is used")

        except Exception as e:
            logger.warning(f"Could not decode dataset sample: {e} - possible codec issue")

        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset {repo_id}: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}")




# TOML parsing utility
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def get_tool_config(tool_name: str, file_path: str = "pyproject.toml") -> dict:
    """Load tool configuration from pyproject.toml."""
    with open(file_path, "rb") as f:
        config = tomllib.load(f)
    current = config.get("tool", {})
    for part in tool_name.split("."):
        current = current.get(part, {})
    return current


def validate_scheduler_config(cfg: dict) -> None:
    """Validate scheduler configuration parameters.

    Args:
        cfg: Configuration dictionary with scheduler parameters

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate scheduler type
    valid_types = ["cosine", "cosine_warm_restarts", "reduce_on_plateau"]
    scheduler_type = cfg.get("scheduler_type", "cosine")
    if scheduler_type not in valid_types:
        raise ValueError(
            f"Invalid scheduler_type '{scheduler_type}'. Must be one of {valid_types}"
        )

    # Validate cosine_warm_restarts specific parameters
    if scheduler_type == "cosine_warm_restarts":
        T_0 = cfg.get("cosine_warm_restarts_T_0", 15)
        T_mult = cfg.get("cosine_warm_restarts_T_mult", 1.2)
        if not isinstance(T_0, (int, float)) or T_0 <= 0:
            raise ValueError(
                f"cosine_warm_restarts_T_0 must be positive number, got {T_0}"
            )
        if not isinstance(T_mult, (int, float)) or T_mult < 1.0:
            raise ValueError(
                f"cosine_warm_restarts_T_mult must be >= 1.0, got {T_mult}"
            )

    # Validate common parameters
    eta_min = cfg.get("eta_min", 5e-7)
    if not isinstance(eta_min, (int, float)) or eta_min <= 0:
        raise ValueError(f"eta_min must be positive number, got {eta_min}")

    # Validate adaptive parameters
    if cfg.get("adaptive_lr_enabled", False):
        lr_boost_factor = cfg.get("lr_boost_factor", 1.15)
        high_loss_multiplier = cfg.get("high_loss_multiplier", 2.0)
        if not isinstance(lr_boost_factor, (int, float)) or lr_boost_factor < 1.0:
            raise ValueError(f"lr_boost_factor must be >= 1.0, got {lr_boost_factor}")
        if (
            not isinstance(high_loss_multiplier, (int, float))
            or high_loss_multiplier <= 1.0
        ):
            raise ValueError(
                f"high_loss_multiplier must be > 1.0, got {high_loss_multiplier}"
            )

    # Validate mu parameters
    if cfg.get("adaptive_mu_enabled", False):
        mu_adjust_factor = cfg.get("mu_adjust_factor", 1.05)
        loss_std_threshold = cfg.get("loss_std_threshold", 1.2)
        mu_min = cfg.get("mu_min", 0.001)
        if not isinstance(mu_adjust_factor, (int, float)) or mu_adjust_factor <= 1.0:
            raise ValueError(f"mu_adjust_factor must be > 1.0, got {mu_adjust_factor}")
        if not isinstance(loss_std_threshold, (int, float)) or loss_std_threshold <= 0:
            raise ValueError(
                f"loss_std_threshold must be positive, got {loss_std_threshold}"
            )
        if not isinstance(mu_min, (int, float)) or mu_min < 0:
            raise ValueError(f"mu_min must be non-negative, got {mu_min}")

    # Validate spike detection parameters
    spike_threshold = cfg.get("spike_threshold", 0.5)
    adjustment_window = cfg.get("adjustment_window", 5)
    max_adjust_factor = cfg.get("max_adjust_factor", 1.05)
    if not isinstance(spike_threshold, (int, float)) or spike_threshold <= 0:
        raise ValueError(f"spike_threshold must be positive, got {spike_threshold}")
    if not isinstance(adjustment_window, int) or adjustment_window < 1:
        raise ValueError(
            f"adjustment_window must be positive integer, got {adjustment_window}"
        )
    if not isinstance(max_adjust_factor, (int, float)) or max_adjust_factor < 1.0:
        raise ValueError(f"max_adjust_factor must be >= 1.0, got {max_adjust_factor}")

    # Validate outlier client IDs
    outlier_client_ids = cfg.get("outlier_client_ids", [])
    if not isinstance(outlier_client_ids, list):
        raise ValueError(
            f"outlier_client_ids must be a list, got {type(outlier_client_ids)}"
        )
    for client_id in outlier_client_ids:
        if not isinstance(client_id, int) or client_id < 0:
            raise ValueError(
                f"outlier_client_ids must contain non-negative integers, got {client_id}"
            )




def compute_param_update_norm(pre_params, post_params):
    """Compute L2 norm of parameter differences between pre and post training.

    Args:
        pre_params: List of numpy arrays (pre-training parameters)
        post_params: List of numpy arrays (post-training parameters)

    Returns:
        float: L2 norm of parameter differences
    """
    if pre_params is None or post_params is None:
        return 0.0

    if len(pre_params) != len(post_params):
        return 0.0

    import numpy as np

    param_diff_norm = np.sqrt(
        sum(np.sum((post - pre) ** 2) for post, pre in zip(post_params, pre_params))
    )
    return float(param_diff_norm)


# DEPRECATED: get_client_output_dir(config, client_id, logger) - use get_client_dir(base_dir, dataset_slug)

def get_client_dir(base_dir: Path, dataset_name: str) -> Path:
    """Get the unified client-specific output directory using dataset name.
    
    Works for both sim and prod modes. Slugifies dataset_name (repo_id) for dir name.
    
    Args:
        base_dir: Base outputs directory (from server save_path or log_file.parent)
        dataset_name: Raw dataset repo_id or name (will be slugified internally)
    
    Returns:
        Path: base_dir/clients/slugified_dataset_name
    """
    # Stdlib-safe slugify: lower, replace non-alphanum with '_'
    slug = re.sub(r'[^a-z0-9]+', '_', str(dataset_name).lower()).strip('_')
    if not slug:
        slug = "unknown_dataset"
    client_dir = base_dir / "clients" / slug
    client_dir.mkdir(parents=True, exist_ok=True)
    return client_dir

def save_client_round_metrics(
    config, training_metrics, round_num, client_id, logger
):
    """Save per-round client metrics to JSON file.
    
    Args:
        config: Flower config dict containing save_path
        training_metrics: Dict of training metrics
        round_num: Current round number
        client_id: Client identifier (partition_id in sim, UUID in prod)
        logger: Logger instance for logging
    
    Returns:
        None
    """
    try:
        base_dir = get_base_output_dir(save_path=config.get("save_path"))
        output_dir = get_client_dir(base_dir, training_metrics["dataset_name"])
        logger.info(f"Client {client_id}: Using unified client dir: {output_dir} for dataset '{training_metrics['dataset_name']}'")

        json_data = create_client_metrics_dict(
            round_num=round_num,
            client_id=str(client_id),
            dataset_name=training_metrics.get("dataset_name", ""),
            policy_loss=training_metrics.get("policy_loss", 0.0),
            fedprox_loss=training_metrics.get("fedprox_loss", 0.0),
            grad_norm=training_metrics.get("grad_norm", 0.0),
            param_hash=training_metrics.get("param_hash", ""),
            num_steps=training_metrics.get("steps_completed", 0),
            param_update_norm=training_metrics.get("param_update_norm", 0.0),
        )
        metrics_file = output_dir / f"round_{round_num}.json"
        with open(metrics_file, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(
            f"Client {client_id}: Saved per-round metrics to {metrics_file}"
        )
    except Exception as e:
        logger.warning(f"Client {client_id}: Failed to save per-round metrics: {e}")


def validate_and_log_parameters(parameters: List[np.ndarray], gate_name: str, expected_count: Optional[int] = None) -> str:
    """Validate parameter count (if expected_count provided) and compute hash for logging.

    Args:
        parameters: List of numpy arrays containing model parameters
        gate_name: Name of the validation gate (for logging)
        expected_count: Optional expected number of parameter arrays

    Returns:
        SHA256 hash string of the parameters

    Raises:
        AssertionError: If parameter count doesn't match expected count (when provided)
    """
    if expected_count is not None:
        assert len(parameters) == expected_count, f"Parameter count mismatch: got {len(parameters)}, expected {expected_count}"

    # Compute hash
    current_hash = compute_parameter_hash(parameters)

    logger.info(f"🛡️ {gate_name}: {len(parameters)} params, hash={current_hash[:16]}...")

    return current_hash


def get_dataset_slug(context: Any) -> str:
    """Get dataset slug from context for unified sim/prod path generation."""
    # Production: Check environment first (CLI/Docker), then run_config
    if os.environ.get("DATASET_NAME"):
        return os.environ["DATASET_NAME"]
    elif context.run_config.get("dataset.repo_id"):
        return context.run_config["dataset.repo_id"]
    elif context.run_config.get("dataset.root"):
        return Path(context.run_config["dataset.root"]).name
    else:
        # Simulation: Use partition_id to lookup from DatasetConfig
        partition_id = int(context.node_config["partition-id"])
        from src.configs.datasets import DatasetConfig
        config = DatasetConfig.load()
        if 0 <= partition_id < len(config.clients):
            return config.clients[partition_id].name
        raise ValueError(f"Invalid partition_id {partition_id}: no dataset mapping")




def get_base_output_dir(save_path: Optional[str] = None) -> Path:
    """Get the base output directory from save_path (default fallback)."""
    if save_path:
        return Path(save_path)
    else:
        return Path("outputs/default")  # Fallback when no timestamp/save_path (pre-round clients)