"""zk0: A Flower / Hugging Face LeRobot app."""

from datetime import datetime
from pathlib import Path

from src.training.model_utils import get_model, get_params

from src.logger import setup_server_logging
from loguru import logger
from .server.strategy import AggregateEvaluationStrategy

from flwr.common import (
    Context,
    ndarrays_to_parameters,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig



def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    import sys

    print("[DEBUG server_fn] Starting server_fn execution", file=sys.stderr)
    sys.stderr.flush()

    try:
        logger.info("🔧 Server: Entering server_fn - starting initialization")
        
        # Create output directory given timestamp (use env var if available, else current time)
        current_time = datetime.now()
        folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")

        # Propagate timestamp to clients for consistent output paths
        context.run_config["timestamp"] = folder_name

        logger.info(f"🔧 Server: Timestamp set to {folder_name}")

        # Set up output directories
        from src.server.server_utils import setup_output_directories
        save_path, clients_dir, server_dir, models_dir, simulation_log_path = setup_output_directories(current_time)
        logger.info(f"🔧 Server: Output directories created at {save_path}")

        # Setup unified logging with loguru
        print(
            f"[DEBUG server_fn] Setting up logging at {simulation_log_path}",
            file=sys.stderr,
        )
        sys.stderr.flush()
        from src.server.server_utils import setup_logging
        setup_logging(simulation_log_path)
        logger.info("🔧 Server: Logging setup completed")

        # DEBUG: Log full context for federation detection
        import pprint
        logger.info(f"🔧 Server: Full context.run_config: {pprint.pformat(dict(context.run_config))}")
        logger.info(f"🔧 Server: Full context.node_config: {pprint.pformat(dict(context.node_config))}")
        logger.info(f"🔧 Server: Full context.state: {pprint.pformat(dict(context.state))}")
        logger.info(f"🔧 Server: Context attributes: {dir(context)}")

        # Construct ServerConfig
        num_rounds = context.run_config["num-server-rounds"]
        config = ServerConfig(num_rounds=num_rounds)

        logger.debug(f"Initialized ServerConfig with {num_rounds} rounds")
        logger.info(f"🔧 Server: Initializing with {num_rounds} rounds")
        logger.debug(f"Created output dir: {save_path}")

        # Log the output directory path when training starts (to console for early visibility)
        logger.info(f"Output directory created: {save_path}")

        # Load environment variables and configuration
        logger.info("🔧 Server: Loading config and environment variables")
        from src.server.server_utils import load_config_and_env
        flwr_config, app_config = load_config_and_env()
        logger.info("🔧 Server: Config and env loaded successfully")

        # Add app-specific configs to context.run_config for strategy access
        context.run_config["checkpoint_interval"] = app_config.get("checkpoint_interval", 2)
        logger.info(f"🔧 Server: Checkpoint interval set to {context.run_config['checkpoint_interval']}")

        # Initialize WandB if enabled
        logger.info("🔧 Server: Initializing WandB")
        from src.server.server_utils import initialize_wandb
        wandb_run, run_id = initialize_wandb(app_config, folder_name, save_path)
        logger.info(f"🔧 Server: WandB initialized with run_id: {run_id}")

        # Store wandb run in context for access by visualization functions
        context.run_config["wandb_run"] = wandb_run

        # Add save_path and log_file_path to run config for clients (for client log paths)
        context.run_config["log_file_path"] = str(simulation_log_path)
        context.run_config["save_path"] = str(save_path)
        context.run_config["wandb_run_id"] = (
            run_id  # Pass shared run_id to clients for unified logging
        )
        logger.info("🔧 Server: Run config updated with paths and WandB ID")

        # Get project version using standard importlib.metadata approach
        logger.info("🔧 Server: Loading project version")
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
        logger.info(f"🔧 Server: Project version determined: {project_version}")

        # Save configuration snapshot
        logger.info("🔧 Server: Saving config snapshot")
        from src.server.server_utils import save_config_snapshot
        save_config_snapshot(context, save_path, current_time, project_version)
        logger.info("🔧 Server: Config snapshot saved")

        # Initialize global model
        logger.info("🔧 Server: Initializing global model")
        from src.server.server_utils import initialize_global_model
        global_model_init, dataset_meta = initialize_global_model()
        logger.info("🔧 Server: Global model initialized successfully")

        # Create strategy
        logger.info("🔧 Server: Creating strategy")
        from src.server.server_utils import create_strategy
        strategy = create_strategy(context, global_model_init, server_dir, models_dir, simulation_log_path, save_path, wandb_run)
        logger.info("🔧 Server: Strategy created successfully")

        logger.debug("Returning ServerAppComponents")
        logger.info("🔧 Server: server_fn completed successfully")

        return ServerAppComponents(config=config, strategy=strategy)

    except Exception as e:
        import traceback
        logger.exception(f"CRITICAL: Exception in server_fn - {str(e)}")
        logger.error(f"Full traceback from server_fn:\n{traceback.format_exc()}")
        raise RuntimeError(f"ServerApp initialization failed: {str(e)}") from e


app = ServerApp(server_fn=server_fn)
