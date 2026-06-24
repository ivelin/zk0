"""zk0 Models: pluggable adapters for LeRobot-compatible models (VLAs, world models/WAMs, etc.).

Phase 0 skeleton for multi-model FL Arena support.
All models use the same LeRobotDataset + ds_meta harness for DRY/MECE.
"""

from .base import BaseModelAdapter  # noqa: F401
from .smolvla_adapter import SmolVLAAdapter  # noqa: F401
from .world_model_adapter import WorldModelAdapter  # noqa: F401

# Minimal Phase 0 registry (used by future get_adapter(model_type))
ADAPTER_REGISTRY = {
    "smolvla": SmolVLAAdapter,
    "world_model": WorldModelAdapter,
}

def get_adapter(model_type: str = "smolvla"):
    """Return adapter class for the given model type (Phase 0)."""
    if model_type not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Registered: {list(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[model_type]()
