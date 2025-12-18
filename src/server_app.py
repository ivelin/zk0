"""zk0: A Flower / Hugging Face LeRobot app."""



from loguru import logger

from flwr.common import (
    Context,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig



def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    import sys
    import traceback

    print("[DEBUG server_fn] 0: server_fn ENTRY - Context type OK", file=sys.stderr)
    sys.stderr.flush()

    try:
        print("[DEBUG server_fn] 1: sys imported OK", file=sys.stderr)
        sys.stderr.flush()

        print("[DEBUG server_fn] 2: Before logger.info - logger available?", file=sys.stderr)
        sys.stderr.flush()
        logger.info("🔧 Server: Entering server_fn - starting initialization")
        print("[DEBUG server_fn] 3: logger.info worked", file=sys.stderr)
        sys.stderr.flush()
        sys.stderr.flush()
        
        print("[DEBUG server_fn] 4: Before datetime import", file=sys.stderr)
        sys.stderr.flush()

        from datetime import datetime
        print("[DEBUG server_fn] 5: datetime imported OK", file=sys.stderr)
        sys.stderr.flush()

        # Create output directory given timestamp (use env var if available, else current time)
        current_time = datetime.now()
        print("[DEBUG server_fn] 6: datetime.now() OK", file=sys.stderr)
        sys.stderr.flush()
        folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        print(f"[DEBUG server_fn] 7: folder_name={folder_name}", file=sys.stderr)
        sys.stderr.flush()

        # Propagate timestamp to clients for consistent output paths
        context.run_config["timestamp"] = folder_name
        print("[DEBUG server_fn] 8: timestamp propagated OK", file=sys.stderr)
        sys.stderr.flush()

        logger.info(f"🔧 Server: Timestamp set to {folder_name}")
        print("[DEBUG server_fn] 9: logger timestamp OK", file=sys.stderr)

        print("[DEBUG server_fn] 10: Before setup_output_directories import", file=sys.stderr)
        sys.stderr.flush()
        from src.server.server_utils import setup_output_directories
        print("[DEBUG server_fn] 11: server_utils.setup_output_directories imported OK", file=sys.stderr)
        sys.stderr.flush()

        print("[DEBUG server_fn] 12: Calling setup_output_directories", file=sys.stderr)
        sys.stderr.flush()
        # Set up output directories
        save_path, clients_dir, server_dir, models_dir, simulation_log_path = setup_output_directories(current_time)
        print(f"[DEBUG server_fn] 13: After setup_output_directories: save_path={save_path}, log_path={simulation_log_path}", file=sys.stderr)
        sys.stderr.flush()
        logger.info(f"🔧 Server: Output directories created at {save_path}")
        print("[DEBUG server_fn] 14: logger output dirs OK", file=sys.stderr)
        sys.stderr.flush()

        # Setup unified logging with loguru
        print(f"[DEBUG server_fn] 15: Setting up logging at {simulation_log_path}", file=sys.stderr)
        sys.stderr.flush()

        print("[DEBUG server_fn] 16: Before setup_logging import", file=sys.stderr)
        sys.stderr.flush()
        from src.server.server_utils import setup_logging
        print("[DEBUG server_fn] 17: setup_logging imported OK", file=sys.stderr)
        sys.stderr.flush()

        print("[DEBUG server_fn] 18: Calling setup_logging", file=sys.stderr)
        sys.stderr.flush()
        setup_logging(simulation_log_path)
        print("[DEBUG server_fn] 19: setup_logging call survived", file=sys.stderr)
        sys.stderr.flush()
        logger.info("🔧 Server: Logging setup completed")
        print("[DEBUG server_fn] 20: logger setup complete OK", file=sys.stderr)
        sys.stderr.flush()

        print("[DEBUG server_fn] 21: Before pprint import", file=sys.stderr)
        sys.stderr.flush()
        import pprint
        print("[DEBUG server_fn] 22: pprint imported OK", file=sys.stderr)
        sys.stderr.flush()

        # DEBUG: Log full context for federation detection
        logger.info(f"🔧 Server: Full context.run_config: {pprint.pformat(dict(context.run_config))}")
        print("[DEBUG server_fn] 23: context.run_config logged OK", file=sys.stderr)
        sys.stderr.flush()

        logger.info(f"🔧 Server: Full context.node_config: {pprint.pformat(dict(context.node_config))}")
        print("[DEBUG server_fn] 24: context.node_config logged OK", file=sys.stderr)
        sys.stderr.flush()

        logger.info(f"🔧 Server: Full context.state: {pprint.pformat(dict(context.state))}")
        print("[DEBUG server_fn] 25: context.state logged OK", file=sys.stderr)
        sys.stderr.flush()

        logger.info(f"🔧 Server: Context attributes: {dir(context)}")
        print("[DEBUG server_fn] 26: context.dir logged OK", file=sys.stderr)
        sys.stderr.flush()

        # Construct ServerConfig
        num_rounds = context.run_config["num-server-rounds"]
        config = ServerConfig(num_rounds=num_rounds)

        logger.debug(f"Initialized ServerConfig with {num_rounds} rounds")
        logger.info(f"🔧 Server: Initializing with {num_rounds} rounds")
        logger.debug(f"Created output dir: {save_path}")

        # Log the output directory path when training starts (to console for early visibility)
        logger.info(f"Output directory created: {save_path}")

        print("[DEBUG server_fn] 27: Before load_config_and_env import", file=sys.stderr)
        sys.stderr.flush()
        from src.server.server_utils import load_config_and_env
        print("[DEBUG server_fn] 28: load_config_and_env imported OK", file=sys.stderr)
        sys.stderr.flush()

        print("[DEBUG server_fn] 29: Calling load_config_and_env", file=sys.stderr)
        sys.stderr.flush()
        # Load environment variables and configuration
        logger.info("🔧 Server: Loading config and environment variables")
        flwr_config, app_config = load_config_and_env()
        print(f"[DEBUG server_fn] 30: load_config_and_env returned flwr_config keys={list(flwr_config.keys()) if flwr_config else None}", file=sys.stderr)
        sys.stderr.flush()
        logger.info("🔧 Server: Config and env loaded successfully")
        print("[DEBUG server_fn] 31: logger config loaded OK", file=sys.stderr)
        sys.stderr.flush()

        # Add app-specific configs to context.run_config for strategy access
        context.run_config["checkpoint_interval"] = app_config.get("checkpoint_interval", 2)
        logger.info(f"🔧 Server: Checkpoint interval set to {context.run_config['checkpoint_interval']}")

        print("[DEBUG server_fn] Before initialize_wandb", file=sys.stderr)
        sys.stderr.flush()
        # Initialize WandB if enabled
        logger.info("🔧 Server: Initializing WandB")
        from src.server.server_utils import initialize_wandb
        wandb_run, run_id = initialize_wandb(app_config, folder_name, save_path)
        logger.info(f"🔧 Server: wandb_run={wandb_run} (type={type(wandb_run)}), run_id={run_id}")
        print(f"[DEBUG server_fn] After initialize_wandb: wandb_run={wandb_run} (type={type(wandb_run)}), run_id={run_id}", file=sys.stderr)
        sys.stderr.flush()
        logger.info(f"🔧 Server: WandB initialized with run_id: {run_id}")
        
        # Store wandb_run_id only (str); skip wandb_run object/None (non-serializable for Flower UserConfig)
        if wandb_run is None:
            logger.warning("⚠️ WandB disabled or failed (missing WANDB_API_KEY?): Skipping wandb_run in run_config")
        else:
            logger.warning("⚠️ Skipping wandb_run object in run_config (use wandb_run_id str only for serialization)")
        
        # Add save_path and log_file_path to run config for clients (for client log paths)
        context.run_config["log_file_path"] = str(simulation_log_path)
        context.run_config["save_path"] = str(save_path)
        context.run_config["wandb_run_id"] = run_id  # Pass shared run_id to clients for unified logging
        logger.info("🔧 Server: Run config updated with paths and WandB ID (wandb_run skipped)")

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

        print("[DEBUG server_fn] Before initialize_global_model", file=sys.stderr)
        sys.stderr.flush()
        # Initialize global model
        logger.info("🔧 Server: Initializing global model")
        from src.server.server_utils import initialize_global_model
        global_model_init, dataset_meta = initialize_global_model()
        print("[DEBUG server_fn] After initialize_global_model - model loaded", file=sys.stderr)
        sys.stderr.flush()
        logger.info("🔧 Server: Global model initialized successfully")

        print("[DEBUG server_fn] Before create_strategy", file=sys.stderr)
        sys.stderr.flush()
        # Create strategy
        logger.info("🔧 Server: Creating strategy")
        from src.server.server_utils import create_strategy
        strategy = create_strategy(context, global_model_init, server_dir, models_dir, simulation_log_path, save_path, wandb_run)
        print("[DEBUG server_fn] After create_strategy - strategy ready", file=sys.stderr)
        sys.stderr.flush()
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
