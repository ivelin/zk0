"""Base adapter interface for zk0 models.

Goal (per Robotics Model Arena vision):
- LeRobot-compatible harness: same datasets + evals for any model (SmolVLA, Pi0, Diffusion Policy, World Models / WAMs).
- Standardized I/O: load via ds_meta, compute primary loss, expose trainable params, reset state per round.
- DRY/MECE: one canonical path for data, one for eval/metrics per model type.
- Future: world model adapters will implement next-obs / latent prediction using sequential episodes.

This is the Phase 0 thin hook. SmolVLA remains the initial concrete implementation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseModelAdapter(ABC):
    """Abstract base for a zk0 model (policy or world model).

    Adapters wrap LeRobot policies (or custom dynamics models) and provide a
    uniform interface for the FL client/server/training code.
    """

    name: str = "base"

    @abstractmethod
    def load(self, dataset_meta: Any, **config: Any) -> Any:
        """Load or construct the underlying model using LeRobot dataset metadata.

        Returns the model instance (e.g. policy or world model module) ready for .to(device).
        Must be deterministic given the same ds_meta + config.
        """
        ...

    def get_underlying(self) -> Any:
        """Return the raw model object (for param get/set, forward, etc.)."""
        raise NotImplementedError

    # Optional hooks for Phase 1+ (world models, joint training, custom schedulers)
    def compute_primary_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the main scalar loss for this model (policy flow-match or WM prediction error)."""
        raise NotImplementedError("Subclasses or concrete adapters implement")

    def reset_for_round(self, round_num: int, initial_lr: Optional[float] = None, **cfg: Any) -> None:
        """Called at start of each FL round (e.g. reset schedulers, stats)."""
        pass

    def get_trainable_params(self) -> list:
        """Return trainable parameters in a form suitable for FedProx / exchange (often delegated)."""
        raise NotImplementedError

    # Future: prediction for world models, etc.
    # def predict_next(self, obs, action): ...
