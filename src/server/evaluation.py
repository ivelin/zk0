"""Evaluation utilities for zk0 federated learning server."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from flwr.common import (
    NDArrays,
)
from loguru import logger

from src.training.model_utils import set_params
from src.training.evaluation import test
from src.common.utils import load_lerobot_dataset
from .visualization import SmolVLAVisualizer
from .wandb_utils import log_wandb_metrics
from .metrics_utils import prepare_server_wandb_metrics
from .server_utils import prepare_server_eval_metrics


def evaluate_single_dataset(
    global_parameters: List[np.ndarray],
    dataset_name: str,
    evaldata_id: Optional[int],
    device,
    eval_batches: int,
    load_lerobot_dataset_fn,
    make_policy_fn,
    set_params_fn,
    test_fn,
):
    """Evaluate shared FL parameters on a single dataset.

    Args:
        global_parameters: Shared FL model parameters (numpy arrays)
        dataset_name: Name of the dataset to evaluate
        evaldata_id: Optional evaldata_id for metrics
        device: Device to run evaluation on
        eval_batches: Number of batches to evaluate (0 = all)
        load_lerobot_dataset_fn: Function to load dataset
        make_policy_fn: Function to create policy
        set_params_fn: Function to set parameters
        test_fn: Function to run evaluation

    Returns:
        dict: Dataset evaluation result
    """
    logger.info(
        f"🔍 Server: Evaluating dataset '{dataset_name}' (evaldata_id={evaldata_id})"
    )

    # Load dataset
    dataset = load_lerobot_dataset_fn(dataset_name)
    logger.info(
        f"✅ Server: Dataset '{dataset_name}' loaded successfully (samples: {len(dataset) if hasattr(dataset, '__len__') else 'unknown'})"
    )

    # Create per-dataset policy instance using dataset metadata
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    cfg = SmolVLAConfig()
    cfg.pretrained_path = "lerobot/smolvla_base"
    policy = make_policy_fn(cfg=cfg, ds_meta=dataset.meta)
    logger.info(
        f"✅ Server: Created policy instance for '{dataset_name}' using dataset meta"
    )

    # Set shared FL parameters to this policy instance
    set_params_fn(policy, global_parameters)
    logger.info("✅ Server: Set shared FL parameters to policy instance")

    # Perform evaluation on this dataset-specific policy
    dataset_loss, dataset_num_examples, dataset_metric = test_fn(
        policy, device=device, eval_batches=eval_batches, dataset=dataset
    )
    logger.info(
        f"✅ Server: Dataset '{dataset_name}' evaluation completed - loss={dataset_loss:.4f}, num_examples={dataset_num_examples}"
    )

    # Clean up to prevent VRAM accumulation
    del policy
    torch.cuda.empty_cache()

    return {
        "dataset_name": dataset_name,
        "evaldata_id": evaldata_id,
        "loss": dataset_loss,
        "num_examples": dataset_num_examples,
        "metrics": dataset_metric,
    }


def evaluate_model_on_datasets(
    global_parameters: List[np.ndarray],
    datasets_config: List,
    device,
    eval_batches: int = 0,
):
    """Evaluate shared FL parameters on multiple datasets using per-dataset policy instances.

    For each dataset, create a fresh policy configured for that dataset's meta (camera views),
    set the shared FL parameters, and run evaluation. This mirrors client-side behavior.

    Args:
        global_parameters: Shared FL model parameters (numpy arrays)
        datasets_config: List of dataset configurations from pyproject.toml
        device: Device to run evaluation on
        eval_batches: Number of batches to evaluate per dataset (0 = all)

    Returns:
        tuple: (composite_loss, total_examples, composite_metrics, per_dataset_results)
    """
    from lerobot.policies.factory import make_policy

    dataset_losses = []
    per_dataset_results = []
    total_examples = 0

    for server_config in datasets_config:
        dataset_result = evaluate_single_dataset(
            global_parameters=global_parameters,
            dataset_name=server_config.name,
            evaldata_id=getattr(server_config, "evaldata_id", None),
            device=device,
            eval_batches=eval_batches,
            load_lerobot_dataset_fn=load_lerobot_dataset,
            make_policy_fn=make_policy,
            set_params_fn=set_params,
            test_fn=test,
        )

        dataset_losses.append(dataset_result["loss"])
        total_examples += dataset_result["num_examples"]
        per_dataset_results.append(dataset_result)

    # Compute composite loss (average across datasets)
    if dataset_losses:
        composite_eval_loss = float(np.mean(dataset_losses))
        logger.info(
            f"✅ Server: Composite evaluation completed - average loss={composite_eval_loss:.4f}, total_examples={total_examples}"
        )
        logger.info(
            f"📊 Per-dataset losses: {[f'{loss:.4f}' for loss in dataset_losses]}"
        )
    else:
        composite_eval_loss = 0.0
        logger.warning(
            "⚠️ Server: No dataset losses computed, using 0.0 as composite loss"
        )

    # Create composite metrics
    composite_metrics = {}
    if per_dataset_results:
        # Use first dataset's metrics as base
        composite_metrics.update(per_dataset_results[0]["metrics"])

        # Add per-dataset loss metrics with evaldata_id suffix
        for result in per_dataset_results:
            evaldata_id = result.get("evaldata_id")
            if evaldata_id is not None:
                loss_key = f"loss_evaldata_id_{evaldata_id}"
                composite_metrics[loss_key] = result["loss"]

    composite_metrics["composite_eval_loss"] = composite_eval_loss
    composite_metrics["num_datasets_evaluated"] = len(dataset_losses)
    composite_metrics["server_eval_dataset_results"] = per_dataset_results

    return composite_eval_loss, total_examples, composite_metrics, per_dataset_results


def should_skip_evaluation(server_round: int, eval_frequency: int) -> bool:
    """Check if evaluation should be skipped based on frequency.

    Args:
        server_round: Current server round number
        eval_frequency: How often to perform evaluation (1 = every round)

    Returns:
        bool: True if evaluation should be skipped
    """
    if server_round % eval_frequency != 0:
        logger.info(
            f"ℹ️ Server: Skipping _server_evaluate for round {server_round} (not multiple of eval_frequency={eval_frequency})"
        )
        return True
    return False


def prepare_evaluation_model(
    parameters: NDArrays, device: torch.device, template_model
) -> torch.nn.Module:
    """Prepare model for evaluation by setting parameters and moving to device.

    Args:
        parameters: Model parameters as NDArrays
        device: Target device for evaluation
        template_model: Cached template model instance

    Returns:
        torch.nn.Module: Prepared model ready for evaluation
    """
    logger.info("🔍 Server: Using cached template model for evaluation...")
    model = template_model
    logger.info(
        f"✅ Server: Template model ready (total params: {sum(p.numel() for p in model.parameters())}"
    )

    # Set parameters
    logger.info("🔍 Server: Setting parameters...")

    set_params(model, parameters)
    logger.info("✅ Server: Parameters set successfully")

    # Move model to device
    model = model.to(device)
    logger.info(f"✅ Server: Model moved to device '{device}'")

    return model


def process_evaluation_metrics(
    strategy,
    server_round: int,
    loss: float,
    metrics: dict,
    aggregated_client_metrics: dict,
    individual_client_metrics: list,
) -> None:
    """Process evaluation metrics and update tracking.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number
        loss: Evaluation loss
        metrics: Evaluation metrics dictionary
        aggregated_client_metrics: Aggregated client metrics
        individual_client_metrics: Individual client metrics
    """
    # Track metrics for plotting
    # DEBUG: Log the discrepancy between metrics["policy_loss"] and actual composite loss
    policy_loss_from_metrics = metrics.get("policy_loss", 0.0)
    logger.info(
        f"DEBUG EVAL: Round {server_round} - metrics['policy_loss']={policy_loss_from_metrics:.4f} (from first dataset), but composite_eval_loss (avg across datasets)={loss:.4f}"
    )
    logger.info(
        f"DEBUG EVAL: Full metrics keys: {list(metrics.keys())}, aggregated_client_metrics keys: {list(aggregated_client_metrics.keys())}"
    )

    round_metrics = {
        "round": server_round,
        "num_clients": aggregated_client_metrics.get("num_clients", 0),
        "avg_policy_loss": loss,  # TEMP: Use composite for now, but log the original
        "avg_client_loss": aggregated_client_metrics.get("avg_client_loss", 0.0),
        "param_update_norm": aggregated_client_metrics.get("param_update_norm", 0.0),
        "debug_original_policy_loss": policy_loss_from_metrics,  # Track the original value
    }
    strategy.federated_metrics_history.append(round_metrics)

    # Track server eval losses for dynamic adjustment
    if not hasattr(strategy, "server_eval_losses"):
        strategy.server_eval_losses = []
    strategy.server_eval_losses.append(loss)
    # Keep only last 10 losses to prevent unbounded growth
    if len(strategy.server_eval_losses) > 10:
        strategy.server_eval_losses = strategy.server_eval_losses[-10:]


def log_evaluation_to_wandb(
    strategy,
    server_round: int,
    loss: float,
    metrics: dict,
    aggregated_client_metrics: dict,
    individual_client_metrics: list,
    per_dataset_results: Optional[List[Dict]] = None,
) -> None:
    """Log evaluation results to WandB.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number
        loss: Evaluation loss
        metrics: Evaluation metrics dictionary
        aggregated_client_metrics: Aggregated client metrics
        individual_client_metrics: Individual client metrics
        per_dataset_results: Optional list of per-dataset evaluation results
    """
    if strategy.wandb_run:
        # Use utility function to prepare WandB metrics with same structure as JSON files
        # This ensures WandB metrics structure matches JSON file structure
        wandb_metrics = prepare_server_wandb_metrics(
            server_round=server_round,
            server_loss=loss,
            server_metrics=metrics,
            aggregated_client_metrics=aggregated_client_metrics,
            individual_client_metrics=individual_client_metrics,
            per_eval_dataset_results=per_dataset_results,
        )

        log_wandb_metrics(wandb_metrics, step=server_round)
        logger.debug(
            f"Logged server eval + client metrics to WandB using utility function: {list(wandb_metrics.keys())}"
        )


def save_evaluation_results(
    strategy,
    server_round: int,
    loss: float,
    num_examples: int,
    metrics: dict,
    aggregated_client_metrics: dict,
    individual_client_metrics: list,
) -> None:
    """Save evaluation results to JSON file.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number
        loss: Evaluation loss
        num_examples: Number of examples evaluated
        metrics: Evaluation metrics dictionary
        aggregated_client_metrics: Aggregated client metrics
        individual_client_metrics: Individual client metrics
    """
    if strategy.server_dir:
        import json

        # Use the shared metrics preparation function for consistency
        data = prepare_server_eval_metrics(strategy, server_round)

        server_file = strategy.server_dir / f"round_{server_round}_server_eval.json"
        with open(server_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"✅ Server: Eval results saved to {server_file}")



