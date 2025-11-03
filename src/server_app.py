"""zk0: A Flower / Hugging Face LeRobot app."""

from datetime import datetime
from pathlib import Path

from src.training.model_utils import get_model, get_params, compute_param_norms, set_params

from src.logger import setup_server_logging
from src.common.parameter_utils import compute_parameter_hash
from loguru import logger
from .server.strategy import AggregateEvaluationStrategy
from .server.server_utils import get_runtime_mode

from flwr.common import (
    Context,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy

import torch
from safetensors.torch import save_file
import numpy as np


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    import sys

    print("[DEBUG server_fn] Starting server_fn execution", file=sys.stderr)
    sys.stderr.flush()

    # Determine runtime mode
    mode = get_runtime_mode(context)
    logger.info(f"🔧 Server: Running in {mode} mode")

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    print(
        f"[DEBUG server_fn] Initialized ServerConfig with {num_rounds} rounds",
        file=sys.stderr,
    )
    sys.stderr.flush()

    logger.info(f"🔧 Server: Initializing with {num_rounds} rounds")

    # Create output directory given timestamp (use env var if available, else current time)
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(f"outputs/{folder_name}")
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG server_fn] Created output dir: {save_path}", file=sys.stderr)
    sys.stderr.flush()

    # Create structured output directories
    clients_dir = save_path / "clients"
    server_dir = save_path / "server"
    models_dir = save_path / "models"
    clients_dir.mkdir(exist_ok=True)
    server_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    # Log the output directory path when training starts (to console for early visibility)
    import sys

    print(f"[INFO] Output directory created: {save_path}", file=sys.stderr, flush=True)

    # Setup unified logging with loguru
    simulation_log_path = save_path / "simulation.log"
    print(
        f"[DEBUG server_fn] Setting up logging at {simulation_log_path}",
        file=sys.stderr,
    )
    sys.stderr.flush()
    setup_server_logging(simulation_log_path)
    logger.info("Server logging initialized")
    print("[DEBUG server_fn] Logging setup complete", file=sys.stderr)
    sys.stderr.flush()

    # Load environment variables from .env file
    print("[DEBUG server_fn] Loading .env", file=sys.stderr)
    sys.stderr.flush()
    try:
        from dotenv import load_dotenv

        load_dotenv()
        print("[DEBUG server_fn] .env loaded successfully", file=sys.stderr)
        sys.stderr.flush()
        logger.debug("Environment variables loaded from .env file")
    except ImportError as e:
        print(f"[DEBUG server_fn] .env load failed (ImportError): {e}", file=sys.stderr)
        sys.stderr.flush()
        logger.debug("python-dotenv not available, skipping .env loading")
    except Exception as e:
        print(f"[DEBUG server_fn] .env load failed: {e}", file=sys.stderr)
        sys.stderr.flush()

    # Get wandb configuration from pyproject.toml
    from src.core.utils import get_tool_config

    flwr_config = get_tool_config("flwr", "pyproject.toml")
    app_config = flwr_config.get("app", {}).get("config", {})

    # Add app-specific configs to context.run_config for strategy access
    context.run_config["checkpoint_interval"] = app_config.get("checkpoint_interval", 2)

    # Initialize WandB if enabled
    print("[DEBUG server_fn] Checking WandB config", file=sys.stderr)
    sys.stderr.flush()
    from src.wandb_utils import init_server_wandb

    wandb_run = None
    run_id = f"zk0-sim-fl-run-{folder_name}"
    use_wandb = app_config.get("use-wandb", False)
    print(f"[DEBUG server_fn] use-wandb={use_wandb}", file=sys.stderr)
    sys.stderr.flush()
    if use_wandb:
        try:
            print("[DEBUG server_fn] Initializing WandB", file=sys.stderr)
            sys.stderr.flush()
            wandb_run = init_server_wandb(
                project="zk0",
                run_id=run_id,
                config=dict(app_config),
                dir=str(save_path),
                notes=f"Federated Learning Server - {num_rounds} rounds",
            )
            print("[DEBUG server_fn] WandB initialized successfully", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"[DEBUG server_fn] WandB init failed: {e}", file=sys.stderr)
            sys.stderr.flush()
            wandb_run = None
    else:
        print("[DEBUG server_fn] Skipping WandB (disabled)", file=sys.stderr)
        sys.stderr.flush()

    # Store wandb run in context for access by visualization functions
    context.run_config["wandb_run"] = wandb_run

    # Add save_path and log_file_path to run config for clients (for client log paths)
    context.run_config["log_file_path"] = str(simulation_log_path)
    context.run_config["save_path"] = str(save_path)
    context.run_config["wandb_run_id"] = (
        run_id  # Pass shared run_id to clients for unified logging
    )

    # Save configuration snapshot
    import json

    # Get project version using standard importlib.metadata approach
    try:
        from importlib.metadata import version

        project_version = version("zk0")
        logger.info(f"✅ Server: Project version loaded: {project_version}")
    except Exception as e:
        logger.warning(f"Could not get version via importlib.metadata: {e}")
        # Fallback: read directly from pyproject.toml
        try:
            import tomli

            with open("pyproject.toml", "rb") as f:
                toml_data = tomli.load(f)
                project_version = toml_data["project"]["version"]
                logger.info(
                    f"✅ Server: Project version loaded via tomli: {project_version}"
                )
        except Exception as fallback_e:
            logger.warning(f"tomli version reading also failed: {fallback_e}")
            project_version = "unknown"

    config_snapshot = {
        "timestamp": current_time.isoformat(),
        "run_config": dict(context.run_config),
        "federation": context.run_config.get("federation", "default"),
        "project_version": project_version,
        "output_structure": {
            "base_dir": str(save_path),
            "simulation_log": str(simulation_log_path),
            "config_file": str(save_path / "config.json"),
            "clients_dir": str(clients_dir),
            "server_dir": str(server_dir),
            "models_dir": str(models_dir),
        },
    }
    with open(save_path / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2, default=str)

    # Set global model initialization
    print("[DEBUG server_fn] Loading DatasetConfig", file=sys.stderr)
    sys.stderr.flush()
    # Load a minimal dataset to get metadata for SmolVLA initialization
    from src.core.utils import load_lerobot_dataset
    from src.configs import DatasetConfig

    try:
        dataset_config = DatasetConfig.load()
        print("[DEBUG server_fn] DatasetConfig loaded", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(f"[DEBUG server_fn] DatasetConfig.load failed: {e}", file=sys.stderr)
        sys.stderr.flush()
        raise

    if not dataset_config.server:
        print(
            "[DEBUG server_fn] No server datasets configured - aborting",
            file=sys.stderr,
        )
        sys.stderr.flush()
        raise ValueError("No server evaluation dataset configured")

    server_config = dataset_config.server[
        0
    ]  # Use server dataset for consistent initialization
    print(
        f"[DEBUG server_fn] Loading server dataset: {server_config.name}",
        file=sys.stderr,
    )
    sys.stderr.flush()

    try:
        dataset = load_lerobot_dataset(server_config.name)
        print("[DEBUG server_fn] Dataset loaded successfully", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(
            f"[DEBUG server_fn] load_lerobot_dataset failed for {server_config.name}: {e}",
            file=sys.stderr,
        )
        sys.stderr.flush()
        raise

    dataset_meta = dataset.meta
    print("[DEBUG server_fn] Getting initial model params", file=sys.stderr)
    sys.stderr.flush()

    try:
        ndarrays = get_params(get_model(dataset_meta=dataset_meta))
        print(
            f"[DEBUG server_fn] Initial params obtained: {len(ndarrays)} arrays",
            file=sys.stderr,
        )
        sys.stderr.flush()
    except Exception as e:
        print(f"[DEBUG server_fn] get_params/get_model failed: {e}", file=sys.stderr)
        sys.stderr.flush()
        raise

    # 🛡️ VALIDATE: Server outgoing parameters (initial model)
    from src.common.parameter_utils import validate_and_log_parameters

    validate_and_log_parameters(ndarrays, "server_initial_model")

    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define strategy with evaluation aggregation
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Add evaluation configuration callback to provide save_path to clients
    eval_frequency = context.run_config.get("eval-frequency", 1)
    eval_batches = context.run_config.get("eval_batches", 0)
    logger.info(
        f"Server: Using eval_frequency={eval_frequency}, eval_batches={eval_batches}"
    )

    # FedProx requires proximal_mu parameter - get from config or use default
    from src.core.utils import get_tool_config

    flwr_config = get_tool_config("flwr", "pyproject.toml")
    app_config = flwr_config.get("app", {}).get("config", {})
    proximal_mu = app_config.get("proximal_mu", 0.01)
    logger.info(f"Server: Using proximal_mu={proximal_mu} for FedProx strategy")

    print("[DEBUG server_fn] Creating AggregateEvaluationStrategy", file=sys.stderr)
    sys.stderr.flush()

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
            num_rounds=num_rounds,  # Pass total rounds for chart generation
            wandb_run=wandb_run,  # Pass wandb run for logging
            context=context,  # Pass context for checkpoint configuration
        )
        print("[DEBUG server_fn] Strategy created successfully", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(
            f"[DEBUG server_fn] AggregateEvaluationStrategy creation failed: {e}",
            file=sys.stderr,
        )
        sys.stderr.flush()
        import traceback

        print(
            f"[DEBUG server_fn] Full traceback: {traceback.format_exc()}",
            file=sys.stderr,
        )
        sys.stderr.flush()
        raise

    print("[DEBUG server_fn] Returning ServerAppComponents", file=sys.stderr)
    sys.stderr.flush()

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
