"""Fit configuration utilities for zk0 server strategy."""

import torch
from typing import Dict, List, Tuple

from flwr.common import FitIns, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from loguru import logger


def configure_fit(strategy, server_round: int, parameters, client_manager):
    """Configure the next round of training."""
    # Log CUDA before training round
    logger.info(
        f"Server: Starting training round {server_round} - CUDA available: {torch.cuda.is_available()}"
    )

    logger.info(f"Server: Configuring fit for round {server_round} (mode: {strategy.mode})")

    # Mode-specific configuration
    if strategy.mode == "production":
        # Production mode: External clients, enhanced security
        logger.info("🔒 Production mode: Configuring for external client connections")
    elif strategy.mode == "simulation":
        # Simulation mode: Local Ray clients
        logger.info("🧪 Simulation mode: Configuring for local Ray clients")

    # Get configuration from pyproject.toml
    from src.core.utils import get_tool_config

    flwr_config = get_tool_config("flwr", "pyproject.toml")
    app_config = flwr_config.get("app", {}).get("config", {})

    # Get base config from parent
    config = super(AggregateEvaluationStrategy, strategy).configure_fit(server_round, parameters, client_manager)
    logger.info(f"✅ Server: Base config generated for {len(config)} clients")

    # Monitor client availability
    if len(config) == 0:
        logger.error(
            f"❌ Server: NO CLIENTS AVAILABLE for fit in round {server_round}"
        )
        logger.error(
            "❌ Server: This indicates clients failed to register or crashed during initialization"
        )

    # Log selected clients for this round
    selected_cids = [client_proxy.cid for client_proxy, _ in config]
    logger.info(
        f"Server: Selected clients for fit in round {server_round}: {selected_cids}"
    )

    # Add round number, log_file_path, and save_path for client logging and eval stats
    updated_config = []
    for i, (client_proxy, fit_ins) in enumerate(config):
        logger.debug(f"Server: Configuring client {i} (CID: {client_proxy.cid})")
        updated_fit_config = fit_ins.config.copy()
        updated_fit_config["round"] = server_round
        updated_fit_config["log_file_path"] = str(strategy.log_file)
        updated_fit_config["save_path"] = str(strategy.save_path)
        updated_fit_config["base_save_path"] = str(strategy.save_path)
        # FedProx: Dynamically adjust proximal_mu and LR based on evaluation trends
        from .parameter_validation import compute_fedprox_parameters

        current_mu, current_lr = compute_fedprox_parameters(
            strategy, server_round, app_config
        )

        updated_fit_config["proximal_mu"] = current_mu
        updated_fit_config["initial_lr"] = current_lr

        # Add initial_lr parameter for client-side training
        initial_lr = app_config.get("initial_lr", 1e-3)
        updated_fit_config["initial_lr"] = initial_lr
        logger.debug(
            f"Server: Added initial_lr={initial_lr} to client {client_proxy.cid} config"
        )

        # + Add use_wandb flag for client-side WandB enablement
        use_wandb = app_config.get("use-wandb", False)
        updated_fit_config["use_wandb"] = use_wandb
        logger.debug(
            f"Server: Added use_wandb={use_wandb} to client {client_proxy.cid} config"
        )

        # Add batch_size from pyproject.toml to override client defaults
        batch_size = app_config.get("batch_size", 64)
        updated_fit_config["batch_size"] = batch_size
        logger.debug(
            f"Server: Added batch_size={batch_size} to client {client_proxy.cid} config"
        )

        # WandB run_id not needed in fit config - client already initialized wandb in client_fn

        # 🛡️ VALIDATE: Server outgoing parameters (for training) - with detailed logging
        from src.core.utils import validate_and_log_parameters

        fit_ndarrays = parameters_to_ndarrays(fit_ins.parameters)
        logger.debug(
            f"Server: Pre-serialization params for client {client_proxy.cid}: {len(fit_ndarrays)} arrays"
        )
        for j, ndarray in enumerate(fit_ndarrays[:3]):  # Log first 3
            logger.debug(
                f"  Pre-serial param {j}: shape={ndarray.shape}, dtype={ndarray.dtype}, min={ndarray.min():.4f}, max={ndarray.max():.4f}"
            )
        if len(fit_ndarrays) > 3:
            logger.debug(f"  ... and {len(fit_ndarrays) - 3} more")

        fit_param_hash = validate_and_log_parameters(
            fit_ndarrays, f"server_fit_r{server_round}_client{i}"
        )
        logger.debug(
            f"Server: Computed hash on pre-Flower params: {fit_param_hash}"
        )

        # 🔐 ADD: Include parameter hash in client config for validation
        updated_fit_config["param_hash"] = fit_param_hash

        updated_fit_ins = FitIns(
            parameters=fit_ins.parameters, config=updated_fit_config
        )
        updated_config.append((client_proxy, updated_fit_ins))

    logger.info(f"✅ Server: Fit configuration complete for round {server_round}")
    return updated_config