"""Model checkpointing utilities for zk0 server strategy."""

from typing import Dict

from loguru import logger


def save_and_push_model(strategy, server_round: int, aggregated_parameters, metrics: Dict):
    """Save model checkpoint and conditionally push to Hugging Face Hub."""
    # Get configuration
    from src.common.utils import get_tool_config

    flwr_config = get_tool_config("flwr", "pyproject.toml")
    app_config = flwr_config.get("app", {}).get("config", {})

    # Get intervals and round counts
    checkpoint_interval = app_config.get("checkpoint_interval", 20)
    num_server_rounds = app_config.get("num-server-rounds", 10)

    # Gate local checkpoint saves to reduce disk usage (use same interval as HF push)
    should_save_local = server_round % checkpoint_interval == 0 or server_round == num_server_rounds
    logger.info(
        f"Server: Round {server_round}/{num_server_rounds}, checkpoint_interval={checkpoint_interval}, should_save_local={should_save_local}"
    )

    # Conditionally save local checkpoint
    checkpoint_dir = None
    if should_save_local:
        from .model_utils import save_model_checkpoint

        checkpoint_dir = save_model_checkpoint(
            strategy, aggregated_parameters, server_round
        )
    else:
        logger.info(
            f"Server: Skipping local checkpoint save for round {server_round} (checkpoint_interval={checkpoint_interval}) - disk savings"
        )

    # Early gate for HF Hub push: only push if total rounds >= checkpoint_interval (avoids tiny/debug runs)
    if num_server_rounds < checkpoint_interval:
        logger.info(
            f"Server: Skipping HF Hub push (num_server_rounds={num_server_rounds} < checkpoint_interval={checkpoint_interval}) - tiny run"
        )
        return checkpoint_dir

    # Check if we should push to HF Hub (only if local checkpoint exists and round qualifies)
    should_push = (checkpoint_dir is not None) and (server_round >= checkpoint_interval or server_round == num_server_rounds)
    logger.info(
        f"Server: Round {server_round}/{num_server_rounds}, checkpoint_interval={checkpoint_interval}, should_push={should_push}"
    )

    # Conditionally push to HF Hub
    if should_push and "hf_repo_id" in app_config:
        from .model_utils import push_model_to_hub_enhanced

        repo_id = app_config["hf_repo_id"]
        try:
            push_model_to_hub_enhanced(checkpoint_dir, repo_id)
            logger.info(f"✅ Server: Successfully pushed checkpoint_round_{server_round} to HF Hub: {repo_id}")
        except Exception as push_error:
            logger.warning(f"⚠️ Server: HF Hub push skipped for round {server_round}: {push_error}. Local checkpoint saved.")
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
    from src.common.utils import get_tool_config

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