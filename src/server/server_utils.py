"""Server utility functions for SmolVLA federated learning."""

from __future__ import annotations

from flwr.common import parameters_to_ndarrays

import torch
import numpy as np
from typing import List
from pathlib import Path
from loguru import logger

# Import utils functions at module level for easier testing
from src.utils import get_tool_config


# Parameter validation utilities for federated learning
def compute_parameter_hash(parameters: List[np.ndarray]) -> str:
    """
    Compute SHA256 hash of model parameters for integrity checking.

    Args:
        parameters: List of parameter arrays

    Returns:
        SHA256 hash string of parameter data
    """
    import hashlib

    # Convert parameters to tensors and concatenate (deterministic ordering)
    tensors = [torch.from_numpy(p).flatten() for p in parameters]
    all_params = torch.cat([t for t in sorted(tensors, key=lambda x: x.numel())])
    param_bytes = all_params.numpy().tobytes()

    return hashlib.sha256(param_bytes).hexdigest()


def compute_rounded_hash(ndarrays, precision="float32"):
    """Compute SHA256 hash of NDArrays after rounding to fixed precision.

    This mitigates float drift from Flower's serialization/deserialization
    by rounding to a consistent dtype before hashing.

    Args:
        ndarrays: List of numpy arrays (model parameters)
        precision: Target dtype for rounding ('float32' or 'float16')

    Returns:
        str: SHA256 hex hash of the rounded, flattened arrays
    """
    import numpy as np
    import hashlib

    # Round to fixed dtype for tolerance
    rounded = [arr.astype(getattr(np, precision)) for arr in ndarrays]
    # Flatten and concatenate
    flat = np.concatenate([r.flatten() for r in rounded])
    # Hash the bytes
    return hashlib.sha256(flat.tobytes()).hexdigest()


def validate_and_log_parameters(
    parameters: List[np.ndarray], gate_name: str, expected_count: int = 506
) -> str:
    """
    Validate parameters and log basic information about them.

    Args:
        parameters: Parameter arrays to validate and log
        gate_name: Name of the gate (e.g., "server_initial_model", "client_update")
        expected_count: Expected number of parameters

    Returns:
        Parameter hash for downstream validation

    Raises:
        AssertionError: If validation fails
    """
    # Basic validation
    assert len(parameters) == expected_count, (
        f"Parameter count mismatch at {gate_name}: expected {expected_count}, got {len(parameters)}"
    )

    # Compute hash
    current_hash = compute_parameter_hash(parameters)

    # Log basic information
    logger.info(f"🛡️ {gate_name}: {len(parameters)} params, hash={current_hash[:8]}...")

    return current_hash


def create_client_metrics_dict(
    round_num: int,
    client_id: str,
    dataset_name: str,
    policy_loss: float,
    fedprox_loss: float,
    grad_norm: float,
    param_hash: str,
    num_steps: int,
    param_update_norm: float,
) -> dict:
    """
    Create standardized client metrics dictionary for JSON logging and server aggregation.

    Args:
        round_num: Current federated learning round
        client_id: Client identifier
        dataset_name: Dataset repository ID
        policy_loss: Pure SmolVLA policy loss
        fedprox_loss: FedProx regularization loss
        grad_norm: Gradient norm
        param_hash: SHA256 hash of parameters
        num_steps: Number of training steps completed
        param_update_norm: L2 norm of parameter updates
        latest_local_epochs: Number of local epochs run by this client in the most recent round
        cumulative_rounds_for_dataset: Total federated rounds this client has participated in for its dataset
        cumulative_epochs_for_dataset: Total local epochs accumulated by this client for its dataset across all rounds

    Returns:
        Standardized metrics dict for consistent client/server use
    """
    total_loss = policy_loss + fedprox_loss
    return {
        "round": round_num,
        "client_id": client_id,
        "dataset_name": dataset_name,
        "loss": total_loss,  # Total loss for compatibility
        "policy_loss": policy_loss,  # Pure SmolVLA flow-matching loss
        "fedprox_loss": fedprox_loss,  # FedProx term
        "grad_norm": grad_norm,
        "param_hash": param_hash,
        "num_steps": num_steps,
        "param_update_norm": param_update_norm,
    }


def prepare_server_wandb_metrics(
    server_round: int,
    server_loss: float,
    server_metrics: dict,
    aggregated_client_metrics: dict,
    individual_client_metrics: list,
) -> dict:
    """
    Prepare server metrics for WandB logging using the same structure as JSON files.

    This function creates a flattened metrics dictionary suitable for WandB logging
    that mirrors the structure used in server JSON evaluation files.

    Args:
        server_round: Current server round number
        server_loss: Server evaluation loss
        server_metrics: Server evaluation metrics dictionary
        aggregated_client_metrics: Aggregated client metrics from last round
        individual_client_metrics: List of individual client metrics from last round

    Returns:
        Dictionary of metrics formatted for WandB logging with appropriate prefixes
    """
    wandb_metrics = {}

    # Add server metrics with server_ prefix
    server_prefix = "server_"
    wandb_metrics[f"{server_prefix}round"] = server_round
    wandb_metrics[f"{server_prefix}eval_loss"] = server_loss
    wandb_metrics[f"{server_prefix}eval_policy_loss"] = server_metrics.get(
        "policy_loss", 0.0
    )
    wandb_metrics[f"{server_prefix}eval_successful_batches"] = server_metrics.get(
        "successful_batches", 0
    )
    wandb_metrics[f"{server_prefix}eval_total_batches"] = server_metrics.get(
        "total_batches_processed", 0
    )
    wandb_metrics[f"{server_prefix}eval_total_samples"] = server_metrics.get(
        "total_samples", 0
    )
    wandb_metrics[f"{server_prefix}eval_action_dim"] = server_metrics.get(
        "action_dim", 7
    )

    # Add aggregated client metrics with server_ prefix
    if aggregated_client_metrics:
        for key, value in aggregated_client_metrics.items():
            wandb_metrics[f"{server_prefix}{key}"] = value

    # Add individual client metrics with client_{id}_ prefix
    for client_metric in individual_client_metrics:
        client_id = client_metric.get("client_id", "unknown")
        prefix = f"client_{client_id}_"

        # Add all client metrics except internal Flower fields
        for key, value in client_metric.items():
            if key not in ["flower_proxy_cid"]:  # Exclude internal fields
                wandb_metrics[f"{prefix}{key}"] = value

    return wandb_metrics


def compute_server_param_update_norm(previous_params, current_params):
    """Compute L2 norm of parameter differences between server rounds.

    Args:
        previous_params: Flower Parameters object from previous round
        current_params: Flower Parameters object from current round

    Returns:
        float: L2 norm of parameter differences
    """
    if previous_params is None or current_params is None:
        return 0.0

    import numpy as np

    current_ndarrays = parameters_to_ndarrays(current_params)
    previous_ndarrays = parameters_to_ndarrays(previous_params)

    if len(current_ndarrays) != len(previous_ndarrays):
        return 0.0

    param_diff_norm = np.sqrt(
        sum(np.sum((c - p) ** 2) for c, p in zip(current_ndarrays, previous_ndarrays))
    )
    return float(param_diff_norm)


def aggregate_client_metrics(validated_results):
    """Aggregate client metrics from validated fit results.

    Args:
        validated_results: List of (client_proxy, fit_result) tuples

    Returns:
        dict: Aggregated client metrics
    """
    import numpy as np

    if not validated_results:
        return {
            "avg_client_loss": 0.0,
            "std_client_loss": 0.0,
            "avg_client_proximal_loss": 0.0,
            "avg_client_grad_norm": 0.0,
            "num_clients": 0,
        }

    client_losses = [
        fit_res.metrics.get("loss", 0.0) for _, fit_res in validated_results
    ]
    client_proximal_losses = [
        fit_res.metrics.get("fedprox_loss", 0.0) for _, fit_res in validated_results
    ]
    client_grad_norms = [
        fit_res.metrics.get("grad_norm", 0.0) for _, fit_res in validated_results
    ]

    return {
        "avg_client_loss": float(np.mean(client_losses)) if client_losses else 0.0,
        "std_client_loss": float(np.std(client_losses))
        if len(client_losses) > 1
        else 0.0,
        "avg_client_proximal_loss": float(np.mean(client_proximal_losses))
        if client_proximal_losses
        else 0.0,
        "avg_client_grad_norm": float(np.mean(client_grad_norms))
        if client_grad_norms
        else 0.0,
        "num_clients": len(validated_results),
    }


def collect_individual_client_metrics(validated_results):
    """Collect individual client metrics for detailed reporting.

    Args:
        validated_results: List of (client_proxy, fit_result) tuples

    Returns:
        list: List of individual client metric dictionaries
    """
    client_metrics = []
    for client_proxy, fit_res in validated_results:
        # Use the simple client_id from metrics (0,1,2,3) instead of the long Flower proxy ID
        simple_client_id = fit_res.metrics.get("client_id", client_proxy.cid)
        raw_metrics = fit_res.metrics
        logger.info(
            f"DEBUG SERVER COLLECT: Client {simple_client_id} raw metrics keys: {list(raw_metrics.keys())}, fedprox={raw_metrics.get('fedprox_loss', 'MISSING')}, param_norm={raw_metrics.get('param_update_norm', 'MISSING')}, policy_loss={raw_metrics.get('policy_loss', 'MISSING')}"
        )
        base_metrics = create_client_metrics_dict(
            round_num=0,  # Round will be set in _server_evaluate
            client_id=simple_client_id,
            dataset_name=raw_metrics.get("dataset_name", ""),
            policy_loss=raw_metrics.get("policy_loss", 0.0),
            fedprox_loss=raw_metrics.get("fedprox_loss", 0.0),
            grad_norm=raw_metrics.get("grad_norm", 0.0),
            param_hash=raw_metrics.get("param_hash", ""),
            num_steps=raw_metrics.get("steps_completed", 0),
            param_update_norm=raw_metrics.get("param_update_norm", 0.0),
        )
        # Add Flower-specific field
        base_metrics["flower_proxy_cid"] = client_proxy.cid
        logger.info(
            f"DEBUG SERVER PROCESSED: Client {simple_client_id} final metrics: policy_loss={base_metrics['policy_loss']}, fedprox_loss={base_metrics['fedprox_loss']}, param_update_norm={base_metrics['param_update_norm']}"
        )
        client_metrics.append(base_metrics)
    return client_metrics


def aggregate_and_log_metrics(
    strategy, server_round: int, validated_results, aggregated_parameters
):
    """Aggregate client metrics and compute parameter update norms.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number
        validated_results: List of validated (client_proxy, fit_result) tuples
        aggregated_parameters: Aggregated parameters from this round

    Returns:
        dict: Aggregated client metrics including parameter update norm
    """
    # Aggregate client metrics before calling parent
    aggregated_client_metrics = aggregate_client_metrics(validated_results)

    # Compute parameter update norm if we have previous parameters
    if strategy.previous_parameters is not None and aggregated_parameters is not None:
        param_update_norm = compute_server_param_update_norm(
            strategy.previous_parameters, aggregated_parameters
        )
        aggregated_client_metrics["param_update_norm"] = param_update_norm

    # Store for use in _server_evaluate
    strategy.last_aggregated_metrics = aggregated_client_metrics

    # Store individual client metrics for detailed reporting
    strategy.last_client_metrics = collect_individual_client_metrics(validated_results)

    # Initialize last_client_metrics if not set (for round 0 evaluation)
    if (
        not hasattr(strategy, "last_client_metrics")
        or strategy.last_client_metrics is None
    ):
        strategy.last_client_metrics = []

    return aggregated_client_metrics


def save_and_push_model(strategy, server_round: int, aggregated_parameters):
    """Save model checkpoint and conditionally push to Hugging Face Hub.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number
        aggregated_parameters: Parameters to save/push
    """
    # Save model checkpoint based on checkpoint_interval configuration
    checkpoint_interval = strategy.context.run_config.get("checkpoint_interval", 5)
    checkpoint_saved = False
    if checkpoint_interval > 0 and server_round % checkpoint_interval == 0:
        try:
            logger.info(
                f"💾 Server: Saving model checkpoint for round {server_round} (interval: {checkpoint_interval})"
            )
            checkpoint_dir = save_model_checkpoint(
                strategy, aggregated_parameters, server_round
            )
            logger.info(
                f"✅ Server: Model checkpoint saved successfully for round {server_round}: {checkpoint_dir}"
            )
            checkpoint_saved = True
        except Exception as e:
            logger.error(
                f"❌ Server: Failed to save model checkpoint for round {server_round}: {e}"
            )
            logger.exception("Traceback in save_model_checkpoint")

    # Save final model checkpoint at the end of training (regardless of checkpoint_interval), but avoid double-save
    if strategy.num_rounds and server_round == strategy.num_rounds:
        try:
            # Perform final evaluation for the last round first
            try:
                logger.info(
                    f"🔍 Server: Performing final evaluation for round {server_round}"
                )
                # Convert Parameters to NDArrays for _server_evaluate
                from flwr.common import parameters_to_ndarrays

                aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
                strategy._server_evaluate(server_round, aggregated_ndarrays, {})
            except Exception as e:
                logger.error(
                    f"❌ Server: Failed final evaluation for round {server_round}: {e}"
                )
                logger.exception("Traceback in final _server_evaluate")

            if not checkpoint_saved:
                logger.info(
                    f"💾 Server: Saving final model checkpoint for round {server_round} (end of training, after evaluation)"
                )
                checkpoint_dir = save_model_checkpoint(
                    strategy, aggregated_parameters, server_round
                )
                logger.info(
                    f"✅ Server: Final model checkpoint saved successfully for round {server_round}: {checkpoint_dir}"
                )
                checkpoint_saved = True

            # Push to Hugging Face Hub if configured (always attempt, even if save failed)
            hf_repo_id = strategy.context.run_config.get("hf_repo_id")
            if hf_repo_id:
                if strategy.num_rounds < checkpoint_interval:
                    logger.info(
                        f"ℹ️ Server: Skipping HF Hub push - num_rounds ({strategy.num_rounds}) < checkpoint_interval ({checkpoint_interval})"
                    )
                else:
                    try:
                        logger.info(
                            f"🚀 Server: Pushing final model to Hugging Face Hub: {hf_repo_id}"
                        )
                        push_model_to_hub_enhanced(checkpoint_dir, hf_repo_id)
                        logger.info(
                            "✅ Server: Model pushed to Hugging Face Hub successfully"
                        )
                    except Exception as push_e:
                        logger.error(
                            f"❌ Server: Failed to push final model to Hub: {push_e}"
                        )
                        logger.exception("Traceback in push_model_to_hub")
                        logger.warning(
                            "⚠️ Server: Continuing training despite Hub push failure"
                        )
            else:
                logger.info("ℹ️ Server: No hf_repo_id configured, skipping Hub push")

        except Exception as e:
            logger.error(f"❌ Server: Failed to save final model: {e}")
            # Still attempt Hub push even if checkpoint save failed
            hf_repo_id = strategy.context.run_config.get("hf_repo_id")
            if hf_repo_id:
                if strategy.num_rounds < checkpoint_interval:
                    logger.info(
                        f"ℹ️ Server: Skipping HF Hub push despite checkpoint failure - num_rounds ({strategy.num_rounds}) < checkpoint_interval ({checkpoint_interval})"
                    )
                else:
                    try:
                        logger.info(
                            f"🚀 Server: Attempting Hub push despite checkpoint save failure: {hf_repo_id}"
                        )
                        push_model_to_hub_enhanced(checkpoint_dir, hf_repo_id)
                        logger.info(
                            "✅ Server: Model pushed to Hub successfully despite checkpoint failure"
                        )
                    except Exception as push_e:
                        logger.error(
                            f"❌ Server: Both checkpoint save and Hub push failed: {push_e}"
                        )


def finalize_round_metrics(
    strategy, server_round: int, aggregated_client_metrics, parent_metrics
):
    """Finalize round metrics by merging and adding diagnostics.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number
        aggregated_client_metrics: Client metrics from aggregation
        parent_metrics: Metrics from FedProx parent class

    Returns:
        dict: Final merged metrics dictionary
    """
    # Merge client metrics with parent metrics
    metrics = {**parent_metrics, **aggregated_client_metrics}

    # DIAGNOSIS METRICS: Add current mu, LR, and eval trend to metrics for JSON/WandB logging
    current_mu = (
        strategy.proximal_mu
    )  # Initial mu; actual per-round mu adjusted in configure_fit
    current_lr = strategy.context.run_config.get("initial_lr", "N/A")
    eval_trend = (
        strategy.server_eval_losses[-3:]
        if hasattr(strategy, "server_eval_losses") and strategy.server_eval_losses
        else "N/A (no eval history)"
    )
    metrics["diagnosis_mu"] = current_mu
    metrics["diagnosis_lr"] = current_lr
    metrics["diagnosis_eval_trend"] = str(
        eval_trend
    )  # Convert to string for JSON serialization
    logger.info(
        f"DIAG R{server_round}: Added to metrics - mu={current_mu}, lr={current_lr}, eval_trend={eval_trend}"
    )

    return metrics


def create_model_template():
    """Create a reusable model template for parameter operations using real dataset meta.

    This function abstracts the template model creation logic from AggregateEvaluationStrategy.__init__.
    It tries to load a real dataset first, then falls back to SO-100 compatible meta if datasets are unavailable.

    Returns:
        torch.nn.Module: SmolVLA model template with correct parameter shapes
    """
    try:
        # Try to load real dataset meta (same as server initialization)
        from src.utils import load_lerobot_dataset
        from src.configs import DatasetConfig
        from src.task import get_model

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

        from src.task import get_model

        meta = SO100Meta()
        template_model = get_model(dataset_meta=meta)
        logger.info("✅ Created model template using SO-100 fallback meta")
        return template_model


def push_model_to_hub_enhanced(checkpoint_dir: Path, hf_repo_id: str) -> None:
    """Enhanced push to Hugging Face Hub with model card, git tags, and rich metadata.

    Args:
        checkpoint_dir: Path to checkpoint directory containing all HF files
        hf_repo_id: Hugging Face repository ID
    """
    try:
        logger.info(
            f"🚀 Starting enhanced HF Hub push for {checkpoint_dir.name} to {hf_repo_id}"
        )

        # Validate checkpoint directory contents
        required_files = ["model.safetensors", "config.json", "README.md"]
        missing = [f for f in required_files if not (checkpoint_dir / f).exists()]
        if missing:
            logger.warning(
                f"⚠️ Missing files in {checkpoint_dir}: {missing} - upload may be incomplete"
            )

        # Get round from dir name
        round_num = (
            int(checkpoint_dir.name.split("_")[-1])
            if "round_" in checkpoint_dir.name
            else 0
        )

        # Get HF token from environment
        import os

        hf_token = os.environ.get("HF_TOKEN")
        logger.info(
            f"🔍 HF_TOKEN check: {'Set' if hf_token else 'MISSING (this will cause 403)'}"
        )
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable not found. Required for pushing to Hugging Face Hub."
            )

        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)

        # Create repo if it doesn't exist
        try:
            logger.info(f"🔍 Creating/ensuring repo '{hf_repo_id}' exists...")
            api.create_repo(
                repo_id=hf_repo_id, repo_type="model", exist_ok=True, private=False
            )
            logger.info(f"✅ Repo '{hf_repo_id}' created or already exists")
        except Exception as create_err:
            logger.error(f"❌ Repo creation failed: {create_err}")
            raise

        # Validate repo existence
        try:
            repo_info = api.repo_info(repo_id=hf_repo_id, repo_type="model")
            logger.info(
                f"✅ Repo '{hf_repo_id}' validated: {repo_info.id} (private: {repo_info.private})"
            )
        except Exception as repo_err:
            logger.error(
                f"❌ Repo '{hf_repo_id}' validation failed post-creation: {repo_err}"
            )
            raise

        # Log all files being uploaded
        all_files = list(checkpoint_dir.glob("*"))
        logger.info(
            f"📁 Files to upload ({len(all_files)}): {[f.name for f in all_files]}"
        )

        # Create commit message
        commit_message = f"Upload federated learning checkpoint {checkpoint_dir.name}"

        logger.info(
            f"🔍 Preparing enhanced upload: folder={checkpoint_dir}, repo={hf_repo_id}, commit='{commit_message}'"
        )

        # Clean legacy .bin files from repo before push
        try:
            files = api.list_repo_files(repo_id=hf_repo_id, repo_type="model")
            for file in files:
                if file.filename.endswith("pytorch_model.bin"):
                    api.delete_file(
                        path_in_repo=file.filename,
                        repo_id=hf_repo_id,
                        repo_type="model",
                        commit_message="Remove legacy pytorch_model.bin",
                    )
                    logger.info(f"Deleted legacy {file.filename} from repo")
        except Exception as cleanup_e:
            logger.warning(f"Pre-push cleanup warning (non-fatal): {cleanup_e}")

        # Push entire directory to Hub
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message=commit_message,
        )

        # Add git tag to HF repo
        try:
            api.create_tag(
                repo_id=hf_repo_id,
                tag=f"fl-round-{round_num}",
                message=f"Federated learning checkpoint round {round_num}",
            )
            logger.info(f"🏷️ Added HF tag 'fl-round-{round_num}' to repo")
        except Exception as tag_e:
            logger.warning(f"⚠️ Failed to add HF tag: {tag_e}")

        # Create local git tags
        try:
            from datetime import datetime

            try:
                from importlib.metadata import version
    
                project_version = version("zk0")
            except Exception as e:
                logger.warning(f"Failed to get project version: {e}")
                project_version = "unknown"

            create_git_tags(datetime.now().isoformat(), project_version, round_num)
        except Exception as git_e:
            logger.warning(f"⚠️ Failed to create local git tags: {git_e}")

        logger.info(
            f"🚀 Model pushed to Hugging Face Hub: https://huggingface.co/{hf_repo_id}"
        )
        logger.info(
            f"📊 Enhanced push completed: uploaded {len(all_files)} files from {checkpoint_dir.name}"
        )

    except Exception as e:
        logger.error(f"❌ Failed enhanced HF Hub push for {checkpoint_dir.name}: {e}")
        logger.error(f"🔍 Detailed error type: {type(e).__name__}, args: {e.args}")
        logger.exception("Full traceback in push_model_to_hub_enhanced")
        raise


def create_git_tags(timestamp: str, version: str, server_round: int) -> None:
    """Create git tags for the federated learning run.

    Args:
        timestamp: ISO timestamp string
        version: Project version
        server_round: Final server round number
    """
    try:
        import subprocess

        # Create lightweight tag for the run
        tag_name = f"fl-run-{timestamp.replace('-', '').replace(':', '').replace('+', 'T').replace('.', '-')}-v{version}"
        subprocess.run(["git", "tag", tag_name], check=True, capture_output=True)
        logger.info(f"🏷️ Created local git tag: {tag_name}")

        # Push the tag
        subprocess.run(
            ["git", "push", "origin", tag_name], check=True, capture_output=True
        )
        logger.info(f"🚀 Pushed git tag to origin: {tag_name}")

    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Failed to create/push git tag: {e}")
    except FileNotFoundError:
        logger.warning("⚠️ Git not available, skipping tag creation")


def extract_training_hyperparameters(context, pyproject_config):
    """Extract training hyperparameters from context and pyproject config.

    Args:
        context: Flower Context object
        pyproject_config: Dict from pyproject.toml [tool.flwr.app.config]

    Returns:
        dict: Hyperparameters dict
    """
    hyperparams = {
        "num_server_rounds": context.run_config.get("num-server-rounds", "N/A"),
        "local_epochs": pyproject_config.get("local-epochs", "N/A"),
        "proximal_mu": pyproject_config.get("proximal_mu", "N/A"),
        "initial_lr": pyproject_config.get("initial_lr", "N/A"),
        "batch_size": pyproject_config.get("batch_size", "N/A"),
        "fraction_fit": context.run_config.get("fraction-fit", "N/A"),
        "fraction_evaluate": context.run_config.get("fraction-evaluate", "N/A"),
        "eval_frequency": context.run_config.get("eval-frequency", "N/A"),
        "eval_batches": context.run_config.get("eval_batches", "N/A"),
        "checkpoint_interval": context.run_config.get("checkpoint_interval", "N/A"),
        "dynamic_training_decay": pyproject_config.get("dynamic_training_decay", False),
        "scheduler_type": pyproject_config.get("scheduler_type", "cosine"),
        "adaptive_lr_enabled": pyproject_config.get("adaptive_lr_enabled", False),
        "adaptive_mu_enabled": pyproject_config.get("adaptive_mu_enabled", False),
    }
    return hyperparams


def extract_datasets(pyproject_config, is_simulation=False):
    """Extract train and eval datasets from pyproject config.

    Args:
        pyproject_config: Dict from pyproject.toml [tool.zk0.datasets]
        is_simulation: Bool indicating if running in simulation mode

    Returns:
        tuple: (train_datasets, eval_datasets) lists of dicts
    """
    train_datasets = []
    eval_datasets = []

    # Train datasets (clients)
    clients = pyproject_config.get("clients", [])
    logger.debug(f"Extracting datasets - found {len(clients)} clients in config")
    for client in clients:
        train_datasets.append(
            {
                "name": client.get("name", "N/A"),
                "description": client.get("description", "N/A"),
                "client_id": client.get("client_id", "N/A"),
            }
        )

    # Eval datasets (server)
    server = pyproject_config.get("server", [])
    logger.debug(
        f"Extracting datasets - found {len(server)} server eval datasets in config"
    )
    for eval_ds in server:
        eval_datasets.append(
            {
                "name": eval_ds.get("name", "N/A"),
                "description": eval_ds.get("description", "N/A"),
                "evaldata_id": eval_ds.get("evaldata_id", "N/A"),
            }
        )

    # Add simulation note if applicable
    if is_simulation and train_datasets:
        train_datasets.append(
            {
                "name": "Simulation Mode",
                "description": f"Simulation-based training with {len(clients)} clients",
                "client_id": "sim",
            }
        )

    logger.info(
        f"📊 Extracted datasets - train: {len(train_datasets)} datasets, eval: {len(eval_datasets)} datasets"
    )
    return train_datasets, eval_datasets


def extract_final_metrics(server_dir, num_rounds):
    """Extract final evaluation metrics from server eval JSON.

    Args:
        server_dir: Path to server output directory
        num_rounds: Total number of rounds

    Returns:
        dict: Final metrics dict
    """
    import json

    metrics_file = server_dir / f"round_{num_rounds}_server_eval.json"
    if not metrics_file.exists():
        logger.warning(f"Final metrics file not found: {metrics_file}")
        return {
            "composite_eval_loss": "N/A",
            "aggregated_client_metrics": {},
            "individual_client_metrics": [],
            "server_eval_dataset_results": [],
        }

    try:
        with open(metrics_file, "r") as f:
            data = json.load(f)

        return {
            "composite_eval_loss": data.get("loss", "N/A"),
            "aggregated_client_metrics": data.get("aggregated_client_metrics", {}),
            "individual_client_metrics": data.get("individual_client_metrics", []),
            "server_eval_dataset_results": data.get("server_eval_dataset_results", []),  # Fixed: directly from data, not nested in metrics
        }
    except Exception as e:
        logger.error(f"Failed to parse final metrics: {e}")
        return {
            "composite_eval_loss": "N/A",
            "aggregated_client_metrics": {},
            "individual_client_metrics": [],
            "server_eval_dataset_results": [],
        }


def prepare_server_eval_metrics(strategy, server_round: int):
    """Prepare full server eval metrics dict from strategy for evaluation results.

    This reuses the same logic as save_evaluation_results to create the complete
    server evaluation JSON structure. Used for both saving eval JSON files and
    checkpoint metrics.json.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number

    Returns:
        dict: Full server eval metrics dict matching the JSON file structure
    """
    # Get latest eval loss
    composite_eval_loss = (
        strategy.server_eval_losses[-1]
        if hasattr(strategy, "server_eval_losses") and strategy.server_eval_losses
        else "N/A"
    )

    # Get aggregated client metrics
    aggregated_client_metrics = getattr(strategy, "last_aggregated_metrics", {})

    # Get individual client metrics
    individual_client_metrics = getattr(strategy, "last_client_metrics", [])

    # Get per-dataset results
    per_dataset_results = getattr(strategy, "last_per_dataset_results", [])

    return {
        "composite_eval_loss": composite_eval_loss,
        "aggregated_client_metrics": aggregated_client_metrics,
        "individual_client_metrics": individual_client_metrics,
        "server_eval_dataset_results": per_dataset_results,
        "num_datasets_evaluated": len(per_dataset_results),
    }


def compute_in_memory_insights(strategy):
    """Compute training insights from strategy's in-memory data.

    Args:
        strategy: The AggregateEvaluationStrategy instance

    Returns:
        dict: Insights dict with convergence_trend, avg_client_loss_trend, etc.
    """
    insights = {
        "convergence_trend": "N/A",
        "avg_client_loss_trend": "N/A",
        "param_update_norm_trend": "N/A",
        "lr_mu_adjustments": "N/A",
        "client_participation_rate": "N/A",
        "anomalies": [],
    }

    # Convergence trend from server eval losses
    if hasattr(strategy, "server_eval_losses") and strategy.server_eval_losses:
        losses = strategy.server_eval_losses
        if len(losses) >= 2:
            insights["convergence_trend"] = f"Policy loss: {losses[0]:.4f} → {losses[-1]:.4f}"

    # Client loss trend and param update norm from federated metrics history
    if hasattr(strategy, "federated_metrics_history") and strategy.federated_metrics_history:
        history = strategy.federated_metrics_history
        client_losses = [m.get("avg_client_loss", 0) for m in history]
        param_norms = [m.get("param_update_norm", 0) for m in history]

        if client_losses:
            insights["avg_client_loss_trend"] = f"Started at {client_losses[0]:.4f}, ended at {client_losses[-1]:.4f}"
        if param_norms:
            avg_norm = sum(param_norms) / len(param_norms)
            insights["param_update_norm_trend"] = f"Average {avg_norm:.6f}"

        # Participation rate
        num_clients_list = [m.get("num_clients", 0) for m in history]
        if num_clients_list:
            avg_participation = sum(num_clients_list) / len(num_clients_list)
            insights["client_participation_rate"] = f"Average {avg_participation:.1f} clients per round"

        # Anomalies (dropouts)
        max_clients = max(num_clients_list) if num_clients_list else 0
        rounds = [m.get("round", 0) for m in history]
        dropouts = [
            r for r, nc in zip(rounds, num_clients_list) if nc < max_clients
        ]
        if dropouts:
            insights["anomalies"].append(f"Client dropouts in rounds: {dropouts}")

    # LR/μ adjustments from last aggregated metrics (contains diagnosis_mu, diagnosis_lr)
    last_metrics = getattr(strategy, "last_aggregated_metrics", {})
    diagnosis_mu = last_metrics.get("diagnosis_mu")
    diagnosis_lr = last_metrics.get("diagnosis_lr")

    if diagnosis_mu is not None and diagnosis_lr is not None:
        insights["lr_mu_adjustments"] = f"Final LR: {diagnosis_lr}, μ: {diagnosis_mu}"
    elif diagnosis_mu is not None:
        insights["lr_mu_adjustments"] = f"μ: {diagnosis_mu}"
    elif diagnosis_lr is not None:
        insights["lr_mu_adjustments"] = f"LR: {diagnosis_lr}"

    return insights


def extract_training_insights(server_dir, num_rounds):
    """Extract training insights from federated metrics and policy loss history.

    Args:
        server_dir: Path to server output directory
        num_rounds: Total number of rounds

    Returns:
        dict: Training insights dict
    """
    import json

    insights = {
        "convergence_trend": "N/A",
        "avg_client_loss_trend": "N/A",
        "param_update_norm_trend": "N/A",
        "lr_mu_adjustments": "N/A",
        "client_participation_rate": "N/A",
        "anomalies": [],
    }

    # Try to load federated_metrics.json
    metrics_file = server_dir / "federated_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)

            # Extract trends
            rounds = [m.get("round", 0) for m in metrics_data]
            client_losses = [m.get("avg_client_loss", 0) for m in metrics_data]
            param_norms = [m.get("param_update_norm", 0) for m in metrics_data]

            if client_losses:
                insights["avg_client_loss_trend"] = (
                    f"Started at {client_losses[0]:.4f}, ended at {client_losses[-1]:.4f}"
                )
            if param_norms:
                insights["param_update_norm_trend"] = (
                    f"Average {sum(param_norms) / len(param_norms):.6f}"
                )

            # Participation rate
            num_clients_list = [m.get("num_clients", 0) for m in metrics_data]
            if num_clients_list:
                avg_participation = sum(num_clients_list) / len(num_clients_list)
                insights["client_participation_rate"] = (
                    f"Average {avg_participation:.1f} clients per round"
                )

            # Anomalies (e.g., dropouts)
            max_clients = max(num_clients_list) if num_clients_list else 0
            dropouts = [
                r for r, nc in zip(rounds, num_clients_list) if nc < max_clients
            ]
            if dropouts:
                insights["anomalies"].append(f"Client dropouts in rounds: {dropouts}")

        except Exception as e:
            logger.error(f"Failed to parse federated metrics: {e}")

    # Try to load policy_loss_history.json (correct filename)
    history_file = server_dir / "policy_loss_history.json"
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                history_data = json.load(f)

            losses = [
                v.get("server_policy_loss", 0)
                for v in history_data.values()
                if isinstance(v, dict)
            ]
            if losses:
                insights["convergence_trend"] = (
                    f"Policy loss: {losses[0]:.4f} → {losses[-1]:.4f}"
                )

        except Exception as e:
            logger.error(f"Failed to parse policy loss history: {e}")

    # Extract LR/μ adjustments from server eval JSON (look for diagnosis metrics)
    eval_file = server_dir / f"round_{num_rounds}_server_eval.json"
    if eval_file.exists():
        try:
            with open(eval_file, "r") as f:
                eval_data = json.load(f)

            # Look for diagnosis metrics that might contain LR/μ info
            metrics = eval_data.get("metrics", {})
            diagnosis_mu = metrics.get("diagnosis_mu")
            diagnosis_lr = metrics.get("diagnosis_lr")

            if diagnosis_mu is not None and diagnosis_lr is not None:
                insights["lr_mu_adjustments"] = f"Final LR: {diagnosis_lr}, μ: {diagnosis_mu}"
            elif diagnosis_mu is not None:
                insights["lr_mu_adjustments"] = f"μ: {diagnosis_mu}"
            elif diagnosis_lr is not None:
                insights["lr_mu_adjustments"] = f"LR: {diagnosis_lr}"

        except Exception as e:
            logger.error(f"Failed to parse server eval for LR/μ: {e}")

    return insights


def save_model_checkpoint(strategy, parameters, server_round: int) -> Path:
    """Save model checkpoint to disk as a complete HF-ready directory.

    Creates a full checkpoint directory with model.safetensors, config.json,
    tokenizer files, README.md, and metrics.json for Hugging Face Hub compatibility.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        parameters: Flower Parameters object containing model weights
        server_round: Current server round number
        models_dir: Directory to save the checkpoint

    Returns:
        Path: Path to the created checkpoint directory
    """
    import json
    from datetime import datetime
    from pathlib import Path
    from safetensors.torch import save_file
    from huggingface_hub import snapshot_download
    import shutil

    try:
        logger.info(
            f"💾 Starting checkpoint directory creation for round {server_round}"
        )

        # Create checkpoint directory
        checkpoint_dir = strategy.models_dir / f"checkpoint_round_{server_round}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Convert Flower Parameters to numpy arrays
        from flwr.common import parameters_to_ndarrays

        ndarrays = parameters_to_ndarrays(parameters)

        # Create state dict with proper parameter names using template model
        model = strategy.template_model
        state_dict = {}
        conversion_errors = []
        for i, ((name, original_param), ndarray) in enumerate(
            zip(model.state_dict().items(), ndarrays)
        ):
            try:
                # Convert numpy array back to torch tensor
                import torch

                tensor = torch.from_numpy(ndarray)

                # Always convert to the original dtype to handle dtype drift
                if original_param.dtype != tensor.dtype:
                    logger.debug(
                        f"Converting param {name} from {tensor.dtype} to {original_param.dtype}"
                    )
                    tensor = tensor.to(original_param.dtype)

                # Validate shape matches
                if tensor.shape != original_param.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: {tensor.shape} vs {original_param.shape}"
                    )

                state_dict[name] = tensor

            except Exception as param_e:
                error_msg = f"Failed to convert param {i} ({name}): {param_e}"
                logger.error(f"❌ {error_msg}")
                conversion_errors.append(error_msg)
                # Continue with other parameters

        if conversion_errors:
            logger.warning(
                f"⚠️ {len(conversion_errors)} parameter conversion errors during checkpoint save for round {server_round}"
            )
            if len(conversion_errors) > len(state_dict) * 0.1:  # More than 10% failed
                raise RuntimeError(
                    f"Too many parameter conversion errors ({len(conversion_errors)}/{len(ndarrays)})"
                )

        # Save model in safetensors format
        safetensors_path = checkpoint_dir / "model.safetensors"
        save_file(state_dict, safetensors_path)

        # Save config from template model
        model.config.save_pretrained(checkpoint_dir)

        # Copy base model files (tokenizer, preprocessor, etc.) using HF Hub cache
        try:
            # Use Hugging Face Hub's built-in caching mechanism
            # snapshot_download will use the default HF cache (~/.cache/huggingface/hub)
            # and reuse existing downloads automatically
            base_model_path = snapshot_download("lerobot/smolvla_base")

            # Copy essential files from HF cache to checkpoint dir
            # Note: config.json is already saved from model.config.save_pretrained() above, so skip it
            essential_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "generation_config.json",
                "preprocessor_config.json",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            ]

            copied_count = 0
            for file_name in essential_files:
                src_path = Path(base_model_path) / file_name
                if src_path.exists():
                    shutil.copy2(src_path, checkpoint_dir / file_name)
                    copied_count += 1
                    logger.debug(f"Copied {file_name} from HF cache")

            logger.info(
                f"✅ Copied {copied_count} essential base model files from HF cache"
            )

        except Exception as copy_e:
            logger.warning(f"⚠️ Failed to copy base model files: {copy_e}")

        # Generate README.md with model card
        flwr_config = get_tool_config("flwr", "pyproject.toml")
        app_config = flwr_config.get("app", {}).get("config", {})
        zk0_config = get_tool_config("zk0", "pyproject.toml")

        hyperparams = extract_training_hyperparameters(strategy.context, app_config)
        is_simulation = strategy.context.run_config.get("federation", "").startswith(
            "local-simulation"
        )
        train_datasets, eval_datasets = extract_datasets(
            zk0_config.get("datasets", {}), is_simulation
        )
        metrics = prepare_server_eval_metrics(strategy, server_round)
        insights = compute_in_memory_insights(strategy)

        try:
            from importlib.metadata import version

            project_version = version("zk0")
        except Exception as e:
            logger.warning(f"Failed to get project version: {e}")
            project_version = "unknown"

        other_info = {
            "timestamp": datetime.now().isoformat(),
            "version": project_version,
            "federation": strategy.context.run_config.get("federation"),
        }

        hf_repo_id = strategy.context.run_config.get("hf_repo_id")
        model_card_content = generate_model_card(
            hyperparams,
            train_datasets,
            eval_datasets,
            metrics,
            insights,
            other_info,
            hf_repo_id,
        )

        readme_path = checkpoint_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)

        # Save metrics.json - use the same full server eval structure
        metrics_data = prepare_server_eval_metrics(strategy, server_round)
        metrics_path = checkpoint_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"✅ Full checkpoint directory created: {checkpoint_dir}")
        logger.info(
            f"📊 Saved {len(state_dict)} parameters, config, tokenizer files, README, and metrics"
        )

        return checkpoint_dir

    except Exception as e:
        logger.error(
            f"❌ Failed to create checkpoint directory for round {server_round}: {e}"
        )
        logger.exception("Full traceback in save_model_checkpoint")
        raise


def generate_model_card(
    hyperparams,
    train_datasets,
    eval_datasets,
    metrics,
    insights,
    other_info,
    hf_repo_id=None,
):
    """Generate model card README.md content.

    Args:
        hyperparams: Dict of training hyperparameters
        train_datasets: List of train dataset dicts
        eval_datasets: List of eval dataset dicts
        metrics: Dict of final evaluation metrics
        insights: Dict of training insights
        other_info: Dict with run timestamp, version, etc.

    Returns:
        str: Model card content
    """
    timestamp = other_info.get("timestamp", "N/A")
    version = other_info.get("version", "N/A")

    content_lines = [
        "---",
        "language: en",
        "tags:",
        "- federated-learning",
        "- flower",
        "- smolvla",
        "- robotics",
        "- manipulation",
        "- so-100",
        "library_name: lerobot",
        "license: apache-2.0",
        "---",
        "",
        "# SmolVLA Federated Learning Checkpoint",
        "",
        "This model is a fine-tuned SmolVLA checkpoint trained using federated learning on SO-100 robotics datasets.",
        "",
        "## Training Details",
        "",
        "**Training Type**: Federated Learning (Flower Framework)",
        "**Base Model**: lerobot/smolvla_base",
        "**Timestamp**: {}".format(timestamp),
        "**Version**: {}".format(version),
    ]

    content_lines.extend([
        "",
        "### Hyperparameters",
        "",
        "- **Server Rounds**: {}".format(hyperparams.get("num_server_rounds", "N/A")),
        "- **Local Epochs**: {}".format(hyperparams.get("local_epochs", "N/A")),
        "- **Proximal μ**: {}".format(hyperparams.get("proximal_mu", "N/A")),
        "- **Initial Learning Rate**: {}".format(hyperparams.get("initial_lr", "N/A")),
        "- **Batch Size**: {}".format(hyperparams.get("batch_size", "N/A")),
        "- **Fraction Fit**: {}".format(hyperparams.get("fraction_fit", "N/A")),
        "- **Fraction Evaluate**: {}".format(hyperparams.get("fraction_evaluate", "N/A")),
        "- **Eval Frequency**: {}".format(hyperparams.get("eval_frequency", "N/A")),
        "- **Eval Batches**: {}".format(hyperparams.get("eval_batches", "N/A")),
        "- **Checkpoint Interval**: {}".format(hyperparams.get("checkpoint_interval", "N/A")),
        "- **Dynamic Training Decay**: {}".format(hyperparams.get("dynamic_training_decay", "N/A")),
        "- **Scheduler Type**: {}".format(hyperparams.get("scheduler_type", "N/A")),
        "- **Adaptive LR Enabled**: {}".format(hyperparams.get("adaptive_lr_enabled", "N/A")),
        "- **Adaptive μ Enabled**: {}".format(hyperparams.get("adaptive_mu_enabled", "N/A")),
        "",
        "### Training Datasets",
        "",
    ])

    for ds in train_datasets:
        content_lines.append("- **{}**: {}".format(ds["name"], ds["description"]))

    content_lines.extend([
        "",
        "### Evaluation Datasets",
        "",
    ])

    for ds in eval_datasets:
        content_lines.append("- **{}**: {}".format(ds["name"], ds["description"]))

    content_lines.extend([
        "",
        "### Final Evaluation Metrics",
        "",
        "- **Composite Eval Loss**: {}".format(metrics.get("composite_eval_loss", "N/A")),
        "- **Aggregated Client Metrics**: {}".format(metrics.get("aggregated_client_metrics", {})),
        "- **Individual Client Metrics**: {} clients".format(len(metrics.get("individual_client_metrics", []))),
        "### Per-Dataset Results",
        "",
    ])

    for ds_result in metrics.get("server_eval_dataset_results", []):
        content_lines.append("- **{}**: Loss {:.4f}".format(
            ds_result.get("dataset_name", "Unknown"),
            ds_result.get("loss", "N/A")
        ))

    content_lines.extend([
        "",
        "### Training Insights",
        "",
    ])

    # Only include insights that are not N/A
    convergence_trend = insights.get("convergence_trend", "N/A")
    if convergence_trend != "N/A":
        content_lines.append("- **Convergence Trend**: {}".format(convergence_trend))

    avg_client_loss_trend = insights.get("avg_client_loss_trend", "N/A")
    if avg_client_loss_trend != "N/A":
        content_lines.append("- **Avg Client Loss Trend**: {}".format(avg_client_loss_trend))

    param_update_norm_trend = insights.get("param_update_norm_trend", "N/A")
    if param_update_norm_trend != "N/A":
        content_lines.append("- **Param Update Norm Trend**: {}".format(param_update_norm_trend))

    lr_mu_adjustments = insights.get("lr_mu_adjustments", "N/A")
    if lr_mu_adjustments != "N/A":
        content_lines.append("- **LR/μ Adjustments**: {}".format(lr_mu_adjustments))

    client_participation_rate = insights.get("client_participation_rate", "N/A")
    if client_participation_rate != "N/A":
        content_lines.append("- **Client Participation Rate**: {}".format(client_participation_rate))

    anomalies = insights.get("anomalies", [])
    content_lines.append("- **Anomalies**: {}".format(", ".join(anomalies) or "None detected"))

    content_lines.extend([
        "",
        "## Usage",
        "",
        "This model can be used for robotics manipulation tasks. Load with:",
        "",
        "```python",
        "from lerobot.policies.smolvla import SmolVLAPolicy",
        "",
        "policy = SmolVLAPolicy.from_pretrained(\"{}\")".format(hf_repo_id or "your-model-id"),
        "",
        "### Model Format",
        "- Weights saved in secure safetensors format (model.safetensors).",
        "- Load with: `from safetensors.torch import load_file; state_dict = load_file(\"model.safetensors\")`",
        "- Avoid legacy pytorch_model.bin for security reasons.",
        "",
        "## Limitations",
        "",
        "- Trained on SO-100 datasets only",
        "",
        "## Citation",
        "",
        "If you use this model, please cite:",
        "",
        "```",
        "@misc{zk0-smolvla-fl-2025},",
        "  title={SmolVLA Federated Learning on SO-100},",
        "  author={Kilo Code, Grok AI, ivelin.eth, and contributors},",
        "  year={2025},",
        "  url={https://github.com/ivelin/zk0}",
        "}",
        "```",
    ])

    return "\n".join(content_lines)
