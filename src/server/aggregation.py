"""Aggregation utilities for zk0 server strategy."""

from typing import List, Tuple

from flwr.common import parameters_to_ndarrays
from loguru import logger


def aggregate_parameters(strategy, results: List[Tuple], failures: List[Tuple], server_round: int):
    """Aggregate parameters from client results."""
    # Log aggregation start
    logger.info(f"Server: Aggregating parameters from {len(results)} clients (failures: {len(failures)})")

    if len(results) == 0:
        logger.warning("Server: No client results to aggregate - returning initial parameters")
        return strategy.initial_parameters

    # Use parent class aggregation
    aggregated_parameters = super(AggregateEvaluationStrategy, strategy).aggregate_fit(server_round, results, failures)

    if aggregated_parameters is None:
        logger.error("Server: Parent aggregation returned None - using initial parameters")
        aggregated_parameters = strategy.initial_parameters

    # Log parameter stats
    aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
    logger.info(f"Server: Aggregated {len(aggregated_ndarrays)} parameter arrays")

    # Compute parameter update norm for monitoring
    from src.training.model_utils import compute_param_norms

    param_update_norm = compute_param_norms(aggregated_ndarrays)
    logger.info(f"Server: Parameter update norm: {param_update_norm:.6f}")

    return aggregated_parameters


def aggregate_and_log_metrics(strategy, results: List[Tuple], failures: List[Tuple], server_round: int):
    """Aggregate client metrics and log them."""
    # Collect individual client metrics
    from .metrics_utils import collect_individual_client_metrics

    individual_metrics = collect_individual_client_metrics(results, server_round)

    # Aggregate metrics
    from .metrics_utils import aggregate_client_metrics

    aggregated_metrics = aggregate_client_metrics(individual_metrics)

    # Add parameter update norm to aggregated metrics
    aggregated_metrics["param_update_norm"] = strategy.param_update_norm

    # Log aggregated metrics
    logger.info(f"Server: Aggregated metrics for round {server_round}:")
    for key, value in aggregated_metrics.items():
        logger.info(f"  {key}: {value}")

    # Store for server evaluation
    strategy.last_client_metrics = individual_metrics

    return aggregated_metrics