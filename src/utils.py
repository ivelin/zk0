"""Utility functions for SmolVLA federated learning."""

from __future__ import annotations

import os
import sys
import torch
from typing import Dict, List, Optional
from loguru import logger

# Import LeRobot components
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, FilteredLeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    FilteredLeRobotDataset = None
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as VLAConfig

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Import PEFT components for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    logger.warning("PEFT not available, LoRA functionality disabled")

def load_smolvla_model(model_name: str = "lerobot/smolvla_base", device: str = "auto") -> SmolVLAPolicy:
    """Load SmolVLA model with automatic freezing based on config.

    The model automatically freezes vision encoder and text model based on
    SmolVLAConfig parameters (freeze_vision_encoder=True, train_expert_only=True).

    Args:
        model_name: Hugging Face model name
        device: Device to load model on ('auto', 'cuda', or 'cpu')

    Returns:
        Loaded SmolVLA model with proper freezing applied

    Raises:
        RuntimeError: If model loading fails
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading SmolVLA model: {model_name} on device {device}")

    # Load model - freezing is handled automatically by SmolVLAConfig
    try:
        model = SmolVLAPolicy.from_pretrained(model_name)
        logger.info("Successfully loaded SmolVLA model")
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load SmolVLA model: {e}")
        raise RuntimeError(f"SmolVLA model loading failed: {e}")


def load_lora_policy(config_path: Optional[str] = None, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), peft_config: Optional[dict] = None, dataset_meta: Optional[object] = None) -> torch.nn.Module:
    """Load SmolVLA policy and apply LoRA if enabled.

    Args:
        config_path: Path to VLA config file (e.g., "src/configs/policy/vla.yaml")
        device: Torch device for model placement
        peft_config: PEFT configuration dict with LoRA settings

    Returns:
        SmolVLA policy with LoRA applied if enabled, else full model
    """
    if LoraConfig is None:
        if peft_config and peft_config.get("enabled", False):
            raise RuntimeError("PEFT/LoRA is enabled in configuration but PEFT library is not available. Install PEFT with: pip install peft")
        logger.warning("PEFT not available, falling back to full fine-tuning")
        # Use the same approach as standalone training script
        from lerobot.configs.train import TrainPipelineConfig
    
        try:
            cfg = TrainPipelineConfig.from_pretrained("lerobot/smolvla_base")
            # Override with our local config if needed
            import yaml
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            # Apply any overrides from local config
            if "policy" in config_dict:
                for key, value in config_dict["policy"].items():
                    if hasattr(cfg.policy, key):
                        setattr(cfg.policy, key, value)
        except Exception as e:
            logger.warning(f"Failed to load config from hub: {e}, using local config only")
            # Fallback to local config
            import yaml
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.configs.default import DatasetConfig
            policy_cfg = SmolVLAConfig(**config_dict.get("policy", {}))
            cfg = type('TrainPipelineConfig', (), {
                'policy': policy_cfg,
                'dataset': DatasetConfig(repo_id="lerobot/svla_so100_pickplace")  # Dummy dataset
            })()
    
        logger.info(f"About to call make_policy with cfg.policy type: {type(cfg.policy)}, ds_meta type: {type(dataset_meta)}")
        logger.info(f"Fallback: About to call make_policy with cfg.policy type: {type(cfg.policy)}, ds_meta type: {type(dataset_meta)}")
        policy = make_policy(cfg.policy, ds_meta=dataset_meta, env_cfg=None)
        for param in policy.parameters():
            param.requires_grad = True
        policy.to(device)
        return policy

    # Use LeRobot's default config loading with proper SmolVLA freezing parameters
    from lerobot.configs.train import TrainPipelineConfig

    try:
        cfg = TrainPipelineConfig.from_pretrained("lerobot/smolvla_base")
        # Ensure SmolVLA freezing parameters are set correctly (like dev branch get_model)
        cfg.policy.freeze_vision_encoder = True  # Freeze SigLIP vision encoder
        cfg.policy.train_expert_only = True       # Only train action expert
        logger.info("SmolVLA config updated: freeze_vision_encoder=True, train_expert_only=True")
    except Exception as e:
        logger.warning(f"Failed to load config from hub: {e}, using default config")
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.configs.default import DatasetConfig
        policy_cfg = SmolVLAConfig()
        policy_cfg.freeze_vision_encoder = True  # Freeze SigLIP vision encoder
        policy_cfg.train_expert_only = True       # Only train action expert
        cfg = type('TrainPipelineConfig', (), {
            'policy': policy_cfg,
            'dataset': DatasetConfig(repo_id="lerobot/svla_so100_pickplace")  # Dummy dataset
        })()
        logger.info("SmolVLA config set: freeze_vision_encoder=True, train_expert_only=True")

    policy = make_policy(cfg.policy, ds_meta=dataset_meta, env_cfg=None)

    if peft_config and peft_config.get("enabled", False):
        logger.info("Applying LoRA adapters to SmolVLA action expert only")

        # Apply LoRA only to the action expert (lm_expert) - vision model remains untouched
        expert_model = policy.model.vlm_with_expert.lm_expert

        # Create LoRA config for expert only
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,  # For VLA sequences (HF docs)
            r=peft_config["rank"],  # 16 (arXiv/FlowerTune recommendation for efficiency)
            lora_alpha=peft_config["alpha"],  # 32
            lora_dropout=peft_config["dropout"],  # 0.1
            target_modules=peft_config["target_modules"],  # Attention + MLP layers in expert
            modules_to_save=peft_config["modules_to_save"],  # Custom projections in expert
            inference_mode=False,  # Training
        )

        # Apply LoRA to expert model only
        try:
            peft_expert = get_peft_model(expert_model, lora_cfg)
            policy.model.vlm_with_expert.lm_expert = peft_expert
            logger.info("LoRA applied to action expert only (vision model untouched)")
        except Exception as e:
            logger.error(f"Failed to apply LoRA to action expert: {e}")
            raise RuntimeError(f"LoRA application failed: {e}")

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in policy.parameters())
        logger.info(f"LoRA applied - trainable: {trainable_params:,} ({trainable_params/total_params:.1%}), total: {total_params:,}")
    else:
        # Full fine-tuning fallback
        logger.info("LoRA disabled, enabling full fine-tuning")
        for param in policy.parameters():
            param.requires_grad = True

    policy.to(device)
    return policy


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
            logger.warning(f"Failed to load config from hub: {e}, using default config")
            # Create minimal config if hub loading fails
            from lerobot.configs.default import DatasetConfig
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

            policy_config = SmolVLAConfig()
            policy_config.pretrained_path = "lerobot/smolvla_base"
            policy_config.push_to_hub = False  # Disable hub pushing for federated learning

            cfg = TrainPipelineConfig(
                dataset=DatasetConfig(repo_id=repo_id),
                policy=policy_config
            )

        # Override dataset configuration with our specific dataset
        cfg.dataset.repo_id = repo_id
        cfg.dataset.decode_videos = True  # Enable video decoding for full metadata (episodes)

        # Load dataset using LeRobot factory (same as standalone training)
        dataset = make_dataset(cfg)
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


# TOML parsing utility
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def get_tool_config(tool_name: str, file_path: str = "pyproject.toml") -> dict:
    """Load tool configuration from pyproject.toml."""
    # If file_path is relative, resolve from project root
    if not os.path.isabs(file_path):
        # Get project root (parent of src directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, file_path)

    logger.debug(f"Loading tool config '{tool_name}' from: {file_path}")
    with open(file_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("tool", {}).get(tool_name, {})


def load_vla_config(config_path: str = "src/configs/policy/vla.yaml") -> dict:
    """Load VLA configuration from YAML file."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)