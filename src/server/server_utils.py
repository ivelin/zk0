"""Server utility functions for SmolVLA federated learning."""

from __future__ import annotations


from pathlib import Path
from loguru import logger
import os

# Import utils functions at module level for easier testing
from src.common.utils import get_tool_config, load_env_safe














def create_model_template():
    """Create a reusable model template for parameter operations using real dataset meta.

    This function abstracts the template model creation logic from AggregateEvaluationStrategy.__init__.
    It tries to load a real dataset first, then falls back to SO-100 compatible meta if datasets are unavailable.

    Returns:
        torch.nn.Module: SmolVLA model template with correct parameter shapes
    """
    try:
        # Try to load real dataset meta (same as server initialization)
        from src.core.utils import load_lerobot_dataset
        from src.configs import DatasetConfig
        from src.training.model_utils import get_model

        dataset_config = DatasetConfig.load()
        if dataset_config.server:
            server_config = dataset_config.server[0]
            dataset = load_lerobot_dataset(server_config.name)
            dataset_meta = dataset.meta
            logger.info(
                f"✅ Created model template using real dataset: {server_config.name}"
            )
            return get_model(dataset_meta=dataset_meta)
        else:
            raise ValueError("No server datasets configured")
    except Exception as e:
        logger.warning(
            f"Failed to load real dataset for template: {e} - falling back to SO-100 meta"
        )

        # Fallback to SO-100 compatible meta for standalone use
        class SO100Meta:
            def __init__(self):
                self.action_dim = 7
                self.state_dim = 0
                self.episode_length = 100
                self.stats = {"action": {"mean": [0.0] * 7, "std": [1.0] * 7}}
                self.features = {
                    "observation.image": {"dtype": "uint8", "shape": [3, 480, 640]},
                    "observation.state": {"dtype": "float32", "shape": [0]},
                    "action": {"dtype": "float32", "shape": [7]},
                }
                self.repo_id = "so100-generic"

        from src.training.model_utils import get_model

        meta = SO100Meta()
        template_model = get_model(dataset_meta=meta)
        logger.info("✅ Created model template using SO-100 fallback meta")
        return template_model


def setup_output_directories(current_time):
    """Set up output directories for the federated learning run.

    Args:
        current_time: datetime object for timestamp

    Returns:
        tuple: (save_path, clients_dir, server_dir, models_dir, simulation_log_path)
    """

    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(f"outputs/{folder_name}")
    save_path.mkdir(parents=True, exist_ok=True)

    # Create structured output directories
    clients_dir = save_path / "clients"
    server_dir = save_path / "server"
    models_dir = save_path / "models"
    clients_dir.mkdir(exist_ok=True)
    server_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    simulation_log_path = save_path / "simulation.log"

    return save_path, clients_dir, server_dir, models_dir, simulation_log_path


def setup_logging(simulation_log_path):
    """Set up unified logging with loguru.

    Args:
        simulation_log_path: Path to the simulation log file
    """
    from src.logger import setup_server_logging

    setup_server_logging(simulation_log_path)
    logger.info("Server logging initialized")
    logger.debug("Logging setup complete")


def load_config_and_env():
    """Load environment variables and configuration.

    Returns:
        tuple: (flwr_config, app_config)
    """
    # Load environment variables from .env file
    load_env_safe()

    # Get wandb configuration from pyproject.toml
    try:
        flwr_config = get_tool_config("flwr", "pyproject.toml")
        logger.debug("get_tool_config succeeded")
    except Exception as e:
        logger.error(f"Failed to load flwr config: {e}")
        raise

    app_config = flwr_config.get("app", {}).get("config", {})

    return flwr_config, app_config


def initialize_wandb(app_config, folder_name, save_path):
    """Initialize WandB if enabled.

    Args:
        app_config: Application configuration dict
        folder_name: Timestamp folder name for run ID
        save_path: Path to save directory

    Returns:
        tuple: (wandb_run, run_id)
    """
    from src.server.wandb_utils import init_server_wandb

    wandb_run = None
    run_id = f"zk0-sim-fl-run-{folder_name}"
    use_wandb = app_config.get("use-wandb", False)
    logger.debug(f"use-wandb={use_wandb}")
    if use_wandb:
        try:
            logger.debug("Initializing WandB")
            wandb_run = init_server_wandb(
                project="zk0",
                run_id=run_id,
                config=dict(app_config),
                dir=str(save_path),
                notes=f"Federated Learning Server - {app_config.get('num-server-rounds', 'N/A')} rounds",
            )
            logger.debug("WandB initialized successfully")
        except Exception as e:
            logger.debug(f"WandB init failed: {e}")
            wandb_run = None
    else:
        logger.debug("Skipping WandB (disabled)")

    return wandb_run, run_id


def save_config_snapshot(context, save_path, current_time, project_version):
    """Save configuration snapshot to JSON file.

    Args:
        context: Flower Context object
        save_path: Path to save directory
        current_time: datetime object
        project_version: Project version string
    """
    import json

    config_snapshot = {
        "timestamp": current_time.isoformat(),
        "run_config": dict(context.run_config),
        "federation": context.run_config.get("federation", "default"),
        "project_version": project_version,
        "output_structure": {
            "base_dir": str(save_path),
            "simulation_log": str(save_path / "simulation.log"),
            "config_file": str(save_path / "config.json"),
            "clients_dir": str(save_path / "clients"),
            "server_dir": str(save_path / "server"),
            "models_dir": str(save_path / "models"),
        },
    }
    with open(save_path / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2, default=str)


def initialize_global_model():
    """Initialize the global SmolVLA model.

    Returns:
        tuple: (global_model_init, dataset_meta)
    """
    from flwr.common import ndarrays_to_parameters
    from src.training.model_utils import get_model, get_params
    from src.common.utils import load_lerobot_dataset
    from src.configs import DatasetConfig

    # Load a minimal dataset to get metadata for SmolVLA initialization
    logger.debug("Loading DatasetConfig")
    try:
        dataset_config = DatasetConfig.load()
        logger.debug("DatasetConfig loaded")
    except Exception as e:
        logger.debug(f"DatasetConfig.load failed: {e}")
        raise

    if not dataset_config.server:
        logger.debug("No server datasets configured - aborting")
        raise ValueError("No server evaluation dataset configured")

    server_config = dataset_config.server[0]  # Use server dataset for consistent initialization
    logger.debug(f"Loading server dataset: {server_config.name}")

    try:
        dataset = load_lerobot_dataset(server_config.name)
        logger.debug("Dataset loaded successfully")
    except Exception as e:
        logger.debug(f"load_lerobot_dataset failed for {server_config.name}: {e}")
        raise

    dataset_meta = dataset.meta
    logger.debug("Getting initial model params")

    try:
        ndarrays = get_params(get_model(dataset_meta=dataset_meta))
        logger.debug(f"Initial params obtained: {len(ndarrays)} arrays")
    except Exception as e:
        logger.debug(f"get_params/get_model failed: {e}")
        raise

    # 🛡️ VALIDATE: Server outgoing parameters (initial model)
    from src.common.parameter_utils import validate_and_log_parameters

    validate_and_log_parameters(ndarrays, "server_initial_model")

    global_model_init = ndarrays_to_parameters(ndarrays)

    return global_model_init, dataset_meta


def create_strategy(context, global_model_init, server_dir, models_dir, simulation_log_path, save_path, wandb_run):
    """Create the AggregateEvaluationStrategy.

    Args:
        context: Flower Context object
        global_model_init: Initial model parameters
        server_dir: Server output directory
        models_dir: Models output directory
        simulation_log_path: Path to simulation log
        save_path: Base save directory
        wandb_run: WandB run object

    Returns:
        AggregateEvaluationStrategy: The configured strategy
    """
    from .strategy import AggregateEvaluationStrategy

    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Add evaluation configuration callback to provide save_path to clients
    eval_frequency = context.run_config.get("eval-frequency", 1)
    eval_batches = context.run_config.get("eval_batches", 0)
    logger.info(
        f"Server: Using eval_frequency={eval_frequency}, eval_batches={eval_batches}"
    )

    # FedProx requires proximal_mu parameter - get from config or use default
    from src.common.utils import get_tool_config

    try:
        flwr_config = get_tool_config("flwr", "pyproject.toml")
        logger.debug("get_tool_config succeeded")
    except Exception as e:
        logger.error(f"Failed to load flwr config: {e}")
        raise

    app_config = flwr_config.get("app", {}).get("config", {})
    proximal_mu = app_config.get("proximal_mu", 0.01)
    logger.info(f"Server: Using proximal_mu={proximal_mu} for FedProx strategy")

    logger.debug("Creating AggregateEvaluationStrategy")

    try:
        strategy = AggregateEvaluationStrategy(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            proximal_mu=proximal_mu,  # Required parameter for FedProx
            server_dir=server_dir,
            models_dir=models_dir,
            log_file=simulation_log_path,
            save_path=save_path,
            num_rounds=context.run_config["num-server-rounds"],  # Pass total rounds for chart generation
            wandb_run=wandb_run,  # Pass wandb run for logging
            context=context,  # Pass context for checkpoint configuration
        )
        logger.debug("Strategy created successfully")
    except Exception as e:
        logger.debug(f"AggregateEvaluationStrategy creation failed: {e}")
        import traceback

        logger.debug(f"Full traceback: {traceback.format_exc()}")
        raise

    return strategy


def prepare_server_eval_metrics(strategy, server_round):
    """Prepare server evaluation metrics for JSON logging and model cards.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number

    Returns:
        dict: Structured metrics dict for JSON serialization
    """
    # Get the latest server composite eval loss (average policy loss across all server evaluation datasets)
    server_composite_policy_loss = (
        strategy.server_eval_losses[-1]
        if hasattr(strategy, "server_eval_losses") and strategy.server_eval_losses
        else "N/A"
    )

    # DEBUG: Log server composite policy loss value
    logger.info(
        f"DEBUG PREPARE_METRICS: Round {server_round} - server_composite_policy_loss={server_composite_policy_loss}"
    )

    # Get aggregated client training metrics (from client-side training)
    client_aggregated_training_metrics = (
        strategy.last_aggregated_metrics
        if hasattr(strategy, "last_aggregated_metrics") and strategy.last_aggregated_metrics
        else {}
    )

    # Get individual client training metrics (per-client from training)
    individual_client_training_metrics = (
        strategy.last_client_metrics
        if hasattr(strategy, "last_client_metrics") and strategy.last_client_metrics
        else []
    )

    # Get per-dataset server evaluation results (detailed policy losses per server eval dataset)
    server_per_dataset_eval_results = (
        strategy.last_per_dataset_results
        if hasattr(strategy, "last_per_dataset_results") and strategy.last_per_dataset_results
        else []
    )

    # Count server evaluation datasets processed
    num_server_eval_datasets = len(server_per_dataset_eval_results)

    # DEBUG: Log per-dataset server evaluation losses if available
    if server_per_dataset_eval_results:
        server_per_eval_dataset_policy_losses = [r["loss"] for r in server_per_dataset_eval_results]
        logger.info(
            f"DEBUG PREPARE_METRICS: Round {server_round} - server_per_eval_dataset_policy_losses: {server_per_eval_dataset_policy_losses}, server_composite avg: {sum(server_per_eval_dataset_policy_losses)/len(server_per_eval_dataset_policy_losses):.4f}"
        )

    return {
        "server_composite_eval_loss": server_composite_policy_loss,  # Average policy loss across all server evaluation datasets
        "client_aggregated_training_metrics": client_aggregated_training_metrics,  # Aggregated metrics from client training (e.g., avg_client_loss)
        "individual_client_training_metrics": individual_client_training_metrics,  # Per-client metrics from training (e.g., policy_loss, fedprox_loss)
        "server_per_dataset_eval_results": server_per_dataset_eval_results,  # Detailed results per server evaluation dataset
        "num_server_eval_datasets": num_server_eval_datasets,  # Number of server evaluation datasets used
    }
