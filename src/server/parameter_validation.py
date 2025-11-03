"""Parameter validation utilities for zk0 server strategy."""

from typing import List, Tuple

from flwr.common import parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes
from loguru import logger


def compute_fedprox_parameters(
    strategy, server_round: int, app_config: dict
) -> Tuple[float, float]:
    """Compute FedProx mu and learning rate parameters for the current round.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number
        app_config: Application configuration from pyproject.toml

    Returns:
        tuple: (current_mu, current_lr) for this round
    """
    # FedProx: Dynamically adjust proximal_mu and LR based on evaluation trends
    initial_mu = strategy.proximal_mu
    current_mu = initial_mu
    # Track current LR across rounds (initialize if not set)
    if not hasattr(strategy, "current_lr"):
        strategy.current_lr = app_config.get("initial_lr", 1e-3)
    current_lr = strategy.current_lr

    # Check if dynamic training decay is enabled
    dynamic_training_decay = app_config.get("dynamic_training_decay", False)
    if (
        dynamic_training_decay
        and hasattr(strategy, "server_eval_losses")
        and len(strategy.server_eval_losses) >= 3
    ):
        from src.training.scheduler_utils import compute_joint_adjustment

        current_mu, current_lr, reason = compute_joint_adjustment(
            strategy.server_eval_losses, initial_mu, current_lr
        )
        logger.info(
            f"Server R{server_round}: Dynamic decay mu={current_mu:.6f}, lr={current_lr:.6f} ({reason}, eval_trend={strategy.server_eval_losses[-3:]})"
        )
        # Update tracked LR for next round
        strategy.current_lr = current_lr
    else:
        # Fallback to fixed halving for early rounds or when disabled
        current_mu = initial_mu / (2 ** (server_round // 10))
        logger.info(
            f"Server R{server_round}: Fixed adjust mu={current_mu:.6f} (initial={initial_mu}, factor=2^(server_round//10))"
        )

    return current_mu, current_lr


def validate_client_parameters(
    strategy, results: List[Tuple[ClientProxy, FitRes]]
) -> List[Tuple[ClientProxy, FitRes]]:
    """Validate client parameter hashes and return only validated results.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        results: List of (client_proxy, fit_result) tuples from successful clients

    Returns:
        List of validated (client_proxy, fit_result) tuples, excluding corrupted clients
    """
    from src.core.utils import compute_parameter_hash, compute_rounded_hash

    validated_results = []
    for client_proxy, fit_res in results:
        client_metrics = fit_res.metrics
        if "param_hash" in client_metrics:
            client_hash = client_metrics["param_hash"]

            # Compute hash of client's actual parameters on server side
            client_params = parameters_to_ndarrays(fit_res.parameters)
            server_computed_hash = compute_parameter_hash(client_params)

            # 🔐 ADD: Use rounded hash for drift-resistant validation
            # Use float32 precision for hash (matches transmission dtype, minimal overhead)
            server_computed_hash = compute_rounded_hash(
                client_params, precision="float32"
            )
            logger.debug(
                f"Server: Client {client_proxy.cid} rounded hash: {server_computed_hash}"
            )

            # Compare hashes (use rounded hash for drift resistance)
            if server_computed_hash == client_hash:
                logger.info(
                    f"✅ Server: Client {client_proxy.cid} parameter hash VALIDATED: {client_hash[:8]}..."
                )
                validated_results.append((client_proxy, fit_res))
            else:
                error_msg = f"Parameter hash MISMATCH for client {client_proxy.cid}! Client: {client_hash[:8]}..., Server: {server_computed_hash[:8]}..."
                logger.error(
                    f"❌ Server: {error_msg} - Excluding corrupted client from aggregation"
                )
        else:
            # No hash provided, include but log warning
            logger.warning(
                f"⚠️ Server: Client {client_proxy.cid} provided no parameter hash - including in aggregation"
            )
            validated_results.append((client_proxy, fit_res))

    return validated_results