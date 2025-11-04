"""Model checkpointing utilities for zk0 server strategy."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flwr.common import parameters_to_ndarrays
from loguru import logger


def save_and_push_model(strategy, server_round: int, aggregated_parameters, metrics: Dict):
    """Save model checkpoint and conditionally push to Hugging Face Hub."""
    # Get configuration
    from src.core.utils import get_tool_config

    flwr_config = get_tool_config("flwr", "pyproject.toml")
    app_config = flwr_config.get("app", {}).get("config", {})

    # Check if we should push to HF Hub
    checkpoint_interval = app_config.get("checkpoint_interval", 20)
    num_server_rounds = app_config.get("num-server-rounds", 10)

    should_push = server_round >= checkpoint_interval or server_round == num_server_rounds
    logger.info(
        f"Server: Round {server_round}/{num_server_rounds}, checkpoint_interval={checkpoint_interval}, should_push={should_push}"
    )

    # Always save local checkpoint
    from .model_utils import save_model_checkpoint

    checkpoint_dir = save_model_checkpoint(
        strategy, aggregated_parameters, server_round
    )

    # Conditionally push to HF Hub
    if should_push and "hf_repo_id" in app_config:
        from .model_utils import push_model_to_hub_enhanced

        repo_id = app_config["hf_repo_id"]
        push_model_to_hub_enhanced(checkpoint_dir, repo_id)
    elif should_push:
        logger.info("ℹ️ Server: No hf_repo_id configured, skipping Hub push")
    else:
        logger.info(
            f"Server: Skipping HF Hub push for round {server_round} (checkpoint_interval={checkpoint_interval})"
        )

    return checkpoint_dir


def finalize_round_metrics(strategy, server_round: int, aggregated_parameters, metrics: Dict):
    """Finalize metrics for the round, adding diagnostics and returning final tuple."""
    # Add diagnostics to metrics
    from src.core.utils import get_tool_config

    flwr_config = get_tool_config("flwr", "pyproject.toml")
    app_config = flwr_config.get("app", {}).get("config", {})

    # Add proximal_mu and initial_lr to metrics for tracking
    current_mu = app_config.get("proximal_mu", 0.01)
    current_lr = app_config.get("initial_lr", 1e-3)
    metrics["proximal_mu"] = current_mu
    metrics["initial_lr"] = current_lr

    # Log final metrics
    logger.info(f"✅ Server: Round {server_round} completed")
    logger.info(f"   Metrics: {metrics}")

    # Return final tuple for Flower
    return aggregated_parameters, metrics