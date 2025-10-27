"""Server utility functions for SmolVLA federated learning."""

from __future__ import annotations

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


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


def compute_rounded_hash(ndarrays, precision='float32'):
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


def create_client_metrics_dict(
    round_num: int,
    client_id: str,
    dataset_name: str,
    policy_loss: float,
    fedprox_loss: float,
    grad_norm: float,
    param_hash: str,
    num_steps: int,
    param_update_norm: float,
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
        latest_local_epochs: Number of local epochs run by this client in the most recent round
        cumulative_rounds_for_dataset: Total federated rounds this client has participated in for its dataset
        cumulative_epochs_for_dataset: Total local epochs accumulated by this client for its dataset across all rounds

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
        "param_update_norm": param_update_norm,
    }


def prepare_server_wandb_metrics(
    server_round: int,
    server_loss: float,
    server_metrics: dict,
    aggregated_client_metrics: dict,
    individual_client_metrics: list,
) -> dict:
    """
    Prepare server metrics for WandB logging using the same structure as JSON files.

    This function creates a flattened metrics dictionary suitable for WandB logging
    that mirrors the structure used in server JSON evaluation files.

    Args:
        server_round: Current server round number
        server_loss: Server evaluation loss
        server_metrics: Server evaluation metrics dictionary
        aggregated_client_metrics: Aggregated client metrics from last round
        individual_client_metrics: List of individual client metrics from last round

    Returns:
        Dictionary of metrics formatted for WandB logging with appropriate prefixes
    """
    wandb_metrics = {}

    # Add server metrics with server_ prefix
    server_prefix = "server_"
    wandb_metrics[f"{server_prefix}round"] = server_round
    wandb_metrics[f"{server_prefix}eval_loss"] = server_loss
    wandb_metrics[f"{server_prefix}eval_policy_loss"] = server_metrics.get("policy_loss", 0.0)
    wandb_metrics[f"{server_prefix}eval_successful_batches"] = server_metrics.get("successful_batches", 0)
    wandb_metrics[f"{server_prefix}eval_total_batches"] = server_metrics.get("total_batches_processed", 0)
    wandb_metrics[f"{server_prefix}eval_total_samples"] = server_metrics.get("total_samples", 0)
    wandb_metrics[f"{server_prefix}eval_action_dim"] = server_metrics.get("action_dim", 7)

    # Add aggregated client metrics with server_ prefix
    if aggregated_client_metrics:
        for key, value in aggregated_client_metrics.items():
            wandb_metrics[f"{server_prefix}{key}"] = value

    # Add individual client metrics with client_{id}_ prefix
    for client_metric in individual_client_metrics:
        client_id = client_metric.get("client_id", "unknown")
        prefix = f"client_{client_id}_"

        # Add all client metrics except internal Flower fields
        for key, value in client_metric.items():
            if key not in ["flower_proxy_cid"]:  # Exclude internal fields
                wandb_metrics[f"{prefix}{key}"] = value

    return wandb_metrics


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

    from flwr.common import parameters_to_ndarrays
    import numpy as np

    current_ndarrays = parameters_to_ndarrays(current_params)
    previous_ndarrays = parameters_to_ndarrays(previous_params)

    if len(current_ndarrays) != len(previous_ndarrays):
        return 0.0

    param_diff_norm = np.sqrt(
        sum(np.sum((c - p) ** 2) for c, p in zip(current_ndarrays, previous_ndarrays))
    )
    return float(param_diff_norm)


def aggregate_client_metrics(validated_results):
    """Aggregate client metrics from validated fit results.

    Args:
        validated_results: List of (client_proxy, fit_result) tuples

    Returns:
        dict: Aggregated client metrics
    """
    import numpy as np

    if not validated_results:
        return {
            "avg_client_loss": 0.0,
            "std_client_loss": 0.0,
            "avg_client_proximal_loss": 0.0,
            "avg_client_grad_norm": 0.0,
            "num_clients": 0,
        }

    client_losses = [
        fit_res.metrics.get("loss", 0.0) for _, fit_res in validated_results
    ]
    client_proximal_losses = [
        fit_res.metrics.get("fedprox_loss", 0.0) for _, fit_res in validated_results
    ]
    client_grad_norms = [
        fit_res.metrics.get("grad_norm", 0.0) for _, fit_res in validated_results
    ]

    return {
        "avg_client_loss": float(np.mean(client_losses)) if client_losses else 0.0,
        "std_client_loss": float(np.std(client_losses))
        if len(client_losses) > 1
        else 0.0,
        "avg_client_proximal_loss": float(np.mean(client_proximal_losses))
        if client_proximal_losses
        else 0.0,
        "avg_client_grad_norm": float(np.mean(client_grad_norms))
        if client_grad_norms
        else 0.0,
        "num_clients": len(validated_results),
    }


def collect_individual_client_metrics(validated_results):
    """Collect individual client metrics for detailed reporting.

    Args:
        validated_results: List of (client_proxy, fit_result) tuples

    Returns:
        list: List of individual client metric dictionaries
    """
    client_metrics = []
    for client_proxy, fit_res in validated_results:
        # Use the simple client_id from metrics (0,1,2,3) instead of the long Flower proxy ID
        simple_client_id = fit_res.metrics.get("client_id", client_proxy.cid)
        raw_metrics = fit_res.metrics
        logger.info(
            f"DEBUG SERVER COLLECT: Client {simple_client_id} raw metrics keys: {list(raw_metrics.keys())}, fedprox={raw_metrics.get('fedprox_loss', 'MISSING')}, param_norm={raw_metrics.get('param_update_norm', 'MISSING')}, policy_loss={raw_metrics.get('policy_loss', 'MISSING')}"
        )
        base_metrics = create_client_metrics_dict(
            round_num=0,  # Round will be set in _server_evaluate
            client_id=simple_client_id,
            dataset_name=raw_metrics.get("dataset_name", ""),
            policy_loss=raw_metrics.get("policy_loss", 0.0),
            fedprox_loss=raw_metrics.get("fedprox_loss", 0.0),
            grad_norm=raw_metrics.get("grad_norm", 0.0),
            param_hash=raw_metrics.get("param_hash", ""),
            num_steps=raw_metrics.get("steps_completed", 0),
            param_update_norm=raw_metrics.get("param_update_norm", 0.0),
        )
        # Add Flower-specific field
        base_metrics["flower_proxy_cid"] = client_proxy.cid
        logger.info(
            f"DEBUG SERVER PROCESSED: Client {simple_client_id} final metrics: policy_loss={base_metrics['policy_loss']}, fedprox_loss={base_metrics['fedprox_loss']}, param_update_norm={base_metrics['param_update_norm']}"
        )
        client_metrics.append(base_metrics)
    return client_metrics