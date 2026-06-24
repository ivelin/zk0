"""SmolVLA concrete adapter (initial implementation for Phase 0).

Delegates to the existing LeRobot factory logic in src/training/model_utils.
This keeps 100% backward compatibility while providing the extension point.
"""

from typing import Any

from loguru import logger

from .base import BaseModelAdapter


class SmolVLAAdapter(BaseModelAdapter):
    """Adapter for lerobot/smolvla_base and fine-tunes (flow-matching VLA policy)."""

    name = "smolvla"

    def __init__(self):
        self._model = None

    def load(self, dataset_meta: Any, **config: Any) -> Any:
        """Load using the original factory path (no behavior change)."""
        # Import here to avoid circulars during early bootstrap
        from src.training.model_utils import _load_smolvla_model

        self._model = _load_smolvla_model(dataset_meta)
        logger.info("SmolVLAAdapter: loaded SmolVLA policy via legacy factory")
        return self._model

    def get_underlying(self) -> Any:
        if self._model is None:
            raise RuntimeError("Call load() before get_underlying()")
        return self._model


# Default registration will happen in registry or on first use
