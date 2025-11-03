"""Parameter validation and hashing utilities for federated learning."""

from __future__ import annotations

from flwr.common import parameters_to_ndarrays

import torch
import numpy as np
from typing import List
from loguru import logger


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


def compute_rounded_hash(ndarrays, precision="float32"):
    """Compute SHA256 hash of NDArrays after rounding to fixed precision.

    This mitigates float drift from Flower's serialization/deserialization
    by rounding to a consistent dtype before hashing.

    Args:
        ndarrays: List of numpy arrays (model parameters)
        precision: Target dtype for rounding ('float32' or 'float16')

    Returns:
        str: SHA256 hex hash of the rounded, flattened arrays
    """
    import numpy as np
    import hashlib

    # Round to fixed dtype for tolerance
    rounded = [arr.astype(getattr(np, precision)) for arr in ndarrays]
    # Flatten and concatenate
    flat = np.concatenate([r.flatten() for r in rounded])
    # Hash the bytes
    return hashlib.sha256(flat.tobytes()).hexdigest()


def validate_and_log_parameters(
    parameters: List[np.ndarray], gate_name: str, expected_count: int = 506
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


def compute_server_param_update_norm(previous_params, current_params):
    """Compute L2 norm of parameter differences between server rounds.

    Args:
        previous_params: Flower Parameters object from previous round
        current_params: Flower Parameters object from current round

    Returns:
        float: L2 norm of parameter differences
    """
    if previous_params is None or current_params is None:
        return 0.0

    import numpy as np

    current_ndarrays = parameters_to_ndarrays(current_params)
    previous_ndarrays = parameters_to_ndarrays(previous_params)

    if len(current_ndarrays) != len(previous_ndarrays):
        return 0.0

    param_diff_norm = np.sqrt(
        sum(np.sum((c - p) ** 2) for c, p in zip(current_ndarrays, previous_ndarrays))
    )
    return float(param_diff_norm)