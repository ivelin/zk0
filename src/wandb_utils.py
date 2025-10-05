"""WandB utilities for federated learning logging."""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def init_wandb(project: str = "zk0", name: str = None, group: str = None, config: Dict[str, Any] = None, dir: str = None, notes: str = None) -> Optional[Any]:
    """Initialize WandB run if enabled and API key is available.

    Args:
        project: WandB project name
        name: Run name
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
                "name": name,
                "group": group,
                "config": config or {},
            }
            if dir:
                init_kwargs["dir"] = dir
            if notes:
                init_kwargs["notes"] = notes

            wandb_run = wandb.init(**init_kwargs)

            # Define metrics with meaningful descriptions for clear dashboard
            wandb.define_metric("model_forward_loss", step_metric="training_step", summary="min")
            wandb.define_metric("fedprox_regularization_loss", step_metric="training_step", summary="mean")
            wandb.define_metric("total_training_loss", step_metric="training_step", summary="min", goal="minimize")
            wandb.define_metric("learning_rate", step_metric="training_step", summary="last")
            wandb.define_metric("gradient_norm", step_metric="training_step", summary="mean")
            wandb.define_metric("federated_client_id", step_metric="training_step", summary="last")
            wandb.define_metric("federated_round_number", step_metric="training_step", summary="max")

            # Set detailed descriptions for dashboard clarity
            wandb.run.summary.update({
                "model_forward_loss": "Loss from SmolVLA model forward pass (LeRobot's main loss)",
                "fedprox_regularization_loss": "FedProx proximal regularization term (mu/2 * ||w - w_global||^2)",
                "total_training_loss": "Combined loss for backpropagation (model_forward_loss + fedprox_regularization_loss)",
                "learning_rate": "Current learning rate used by optimizer",
                "gradient_norm": "L2 norm of gradients after clipping",
                "federated_client_id": "Unique identifier for the federated learning client",
                "federated_round_number": "Current federated learning round number"
            })

            logger.info(f"WandB initialized: {wandb_run.name} in project {wandb_run.project}")
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