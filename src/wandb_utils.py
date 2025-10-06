"""WandB utilities for federated learning logging."""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def init_wandb(project: str = "zk0", name: str = None, run_id: str = None, group: str = None, config: Dict[str, Any] = None, dir: str = None, notes: str = None) -> Optional[Any]:
    """Initialize WandB run if enabled and API key is available.

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

    try:
        import wandb

        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
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


def log_wandb_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to WandB if WandB is initialized.

    Args:
        metrics: Dict of metrics to log
        step: Optional step number
    """
    try:
        import wandb
        if wandb.run is not None:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        else:
            logger.debug("WandB run not initialized, skipping metrics logging")
    except ImportError:
        logger.debug("wandb not available, skipping metrics logging")
    except Exception as e:
        logger.warning(f"Failed to log metrics to WandB: {e}")


def init_client_wandb(partition_id: int, dataset_name: str, local_epochs: int, batch_size: int, wandb_run_id: Optional[str] = None, wandb_dir: Optional[str] = None) -> Optional[Any]:
    """Initialize WandB for a federated learning client.

    Args:
        partition_id: Client partition ID
        dataset_name: Dataset repository ID
        local_epochs: Number of local training epochs
        batch_size: Training batch size
        wandb_run_id: Server's WandB run ID to join (if provided)
        wandb_dir: Directory for WandB files

    Returns:
        WandB run object if initialized, None otherwise
    """
    wandb_run = None

    try:
        import wandb

        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            init_kwargs = {
                "project": "zk0",
                "config": {
                    f"client_{partition_id}_id": partition_id,
                    f"client_{partition_id}_dataset": dataset_name,
                    f"client_{partition_id}_local_epochs": local_epochs,
                    f"client_{partition_id}_batch_size": batch_size,
                },
                "notes": f"Federated Learning Client {partition_id} - Dataset: {dataset_name}"
            }

            if wandb_dir:
                init_kwargs["dir"] = wandb_dir

            if wandb_run_id:
                # Join server's unified run
                init_kwargs["id"] = wandb_run_id
                wandb_run = wandb.init(**init_kwargs)
                logger.info(f"Client {partition_id}: Joined unified WandB run {wandb_run.name} (ID: {wandb_run.id})")
            else:
                # Fallback: create separate run (should not happen in normal operation)
                fallback_name = f"client_{partition_id}_{dataset_name}"
                init_kwargs["name"] = fallback_name
                wandb_run = wandb.init(**init_kwargs)
                logger.warning(f"Client {partition_id}: Created separate WandB run {wandb_run.name} (ID: {wandb_run.id}) - no server run_id provided")
        else:
            logger.warning("WANDB_API_KEY not found in environment variables. WandB logging disabled.")

    except ImportError:
        logger.warning("wandb not available. Install with: pip install wandb")
    except Exception as e:
        logger.error(f"Failed to initialize WandB for client {partition_id}: {e}")

    return wandb_run


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