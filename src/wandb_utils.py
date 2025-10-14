"""WandB utilities for federated learning logging."""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def _load_wandb_env_and_log_key(context: str = "") -> Optional[str]:
    """Load environment variables from .env file and log WANDB_API_KEY status.

    Args:
        context: Context string for logging (e.g., "server", "client")

    Returns:
        WANDB_API_KEY if available, None otherwise
    """
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.debug(f"Environment variables loaded from .env file in {context}")
    except ImportError:
        logger.debug(f"python-dotenv not available in {context}, skipping .env loading")

    # Log WANDB_API_KEY status without revealing full key
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        key_preview = wandb_key[-8:] if len(wandb_key) >= 8 else wandb_key
        context_prefix = f" in {context}" if context else ""
        logger.info(f"WANDB_API_KEY loaded{context_prefix}: Yes (ends with ...{key_preview})")
    else:
        context_prefix = f" for {context}" if context else ""
        logger.warning(f"WANDB_API_KEY not found in environment variables{context_prefix}. WandB logging disabled.")

    return wandb_key


def init_server_wandb(project: str = "zk0", name: str = None, run_id: str = None, group: str = None, config: Dict[str, Any] = None, dir: str = None, notes: str = None) -> Optional[Any]:
    """Initialize WandB run for server-side logging if enabled and API key is available.

    Args:
        project: WandB project name
        name: Run name (used to create new run)
        run_id: Run ID (used to join existing run)
        config: Configuration dict to log
        dir: Directory for WandB files
        notes: Run description/notes

    Returns:
        WandB run object if initialized, None otherwise
    """
    wandb_run = None

    wandb_key = _load_wandb_env_and_log_key("server")

    try:
        import wandb

        if wandb_key:
            init_kwargs = {
                "project": project,
                "config": config or {},
            }
            if name:
                init_kwargs["name"] = name
            if run_id:
                init_kwargs["id"] = run_id
            if group:
                init_kwargs["group"] = group
            if dir:
                init_kwargs["dir"] = dir
            if notes:
                init_kwargs["notes"] = notes

            wandb_run = wandb.init(**init_kwargs)

            logger.info(f"WandB initialized: {wandb_run.name} (ID: {wandb_run.id}) in project {wandb_run.project}")
        else:
            logger.warning("WANDB_API_KEY not found in environment variables. WandB logging disabled.")

    except ImportError:
        logger.warning("wandb not available. Install with: pip install wandb")
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")

    return wandb_run


def log_wandb_metrics(metrics: Dict[str, Any], step: Optional[int] = None, description: Optional[str] = None) -> None:
    """Log metrics to WandB if WandB is initialized.

    Args:
        metrics: Dict of metrics to log
        step: Optional step number
        description: Optional description for the metrics (for better chart legends)
    """
    try:
        import wandb
        if wandb.run is not None:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
            if description:
                logger.debug(f"Logged WandB metrics: {description}")
        else:
            logger.debug("WandB run not initialized, skipping metrics logging")
    except ImportError:
        logger.debug("wandb not available, skipping metrics logging")
    except Exception as e:
        logger.warning(f"Failed to log metrics to WandB: {e}")




def finish_wandb() -> None:
    """Finish WandB run if active."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to finish WandB run: {e}")