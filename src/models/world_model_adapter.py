"""World Model (WAM) concrete adapter (minimal stub for Phase 0).

Provides a load-compatible implementation using ds_meta from LeRobotDataset.
This proves the harness (same datasets) can drive both policy (VLA) and
world-model paths without changing existing SmolVLA behavior.

For Phase 0 the implementation is intentionally a thin stub (no dynamics
training logic yet — that is Phase 1). It satisfies the documented
BaseModelAdapter contract so get_model(..., model_type="world_model") works.
"""

from typing import Any

from loguru import logger

from .base import BaseModelAdapter


class WorldModelAdapter(BaseModelAdapter):
    """Minimal adapter for a world model (dynamics predictor).

    In later phases this will load a proper forward model (e.g. latent
    predictor or observation predictor) using the same LeRobotDataset meta
    that VLA policies use.
    """

    name = "world_model"

    def __init__(self):
        self._model = None

    def load(self, dataset_meta: Any, **config: Any) -> Any:
        """Load a stub world-model module using the provided dataset meta.

        Uses the action dim / observation features from ds_meta so that
        the same client datasets that work for SmolVLA also work here.
        """
        # Minimal real module so that downstream code (param get/set,
        # .to(device), state_dict etc.) can treat it like a policy.
        import torch
        import torch.nn as nn

        # Extract basic shape info from LeRobot ds_meta (same as VLA path)
        action_dim = 1
        try:
            if hasattr(dataset_meta, "features") and "action" in dataset_meta.features:
                action_shape = dataset_meta.features["action"].get("shape", [1])
                action_dim = action_shape[0] if isinstance(action_shape, (list, tuple)) else 1
            elif hasattr(dataset_meta, "action_dim"):
                action_dim = int(dataset_meta.action_dim)
        except Exception:
            action_dim = 6  # reasonable default for SO-100 arms

        # Very small dynamics stub: takes (obs_embed + action) -> next_embed
        # In Phase 0 we don't care about real obs embedding; this just proves
        # the load path + get_underlying contract.
        class _StubWorldModel(nn.Module):
            def __init__(self, action_dim: int):
                super().__init__()
                self.action_dim = action_dim
                # tiny linear just to have trainable params and forward
                self.dynamics = nn.Linear(action_dim + 8, 8)  # fake latent dim 8

            def forward(self, action, latent=None):
                if latent is None:
                    latent = torch.zeros(action.shape[0], 8, device=action.device)
                x = torch.cat([latent, action], dim=-1)
                return self.dynamics(x)

        self._model = _StubWorldModel(action_dim)
        logger.info(
            f"WorldModelAdapter: loaded minimal WM stub (action_dim={action_dim}) "
            "using ds_meta — harness compatibility proven for Phase 0"
        )
        return self._model

    def get_underlying(self) -> Any:
        if self._model is None:
            raise RuntimeError("Call load() before get_underlying()")
        return self._model
