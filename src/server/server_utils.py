"""Server utility functions for SmolVLA federated learning."""

from __future__ import annotations

from flwr.common import parameters_to_ndarrays

import torch
import numpy as np
from typing import List
from pathlib import Path
from loguru import logger

# Import utils functions at module level for easier testing
from src.core.utils import get_tool_config
from src.common.parameter_utils import (
    compute_parameter_hash,
    compute_rounded_hash,
    validate_and_log_parameters,
    compute_server_param_update_norm,
)
from src.server.metrics_utils import (
    create_client_metrics_dict,
    prepare_server_wandb_metrics,
    aggregate_client_metrics,
    collect_individual_client_metrics,
    aggregate_and_log_metrics,
    finalize_round_metrics,
)
from src.server.model_utils import (
    push_model_to_hub_enhanced,
    create_git_tags,
    extract_training_hyperparameters,
    extract_datasets,
    extract_final_metrics,
    compute_in_memory_insights,
    extract_training_insights,
    save_model_checkpoint,
    generate_model_card,
)


def get_runtime_mode(context) -> str:
    """Determine runtime mode from context.

    Args:
        context: Flower Context containing run configuration

    Returns:
        str: 'simulation' for local Ray-based runs, 'production' for external client connections
    """
    federation = context.run_config.get("federation", "")
    if "local-simulation" in federation:
        return "simulation"
    else:
        return "production"












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


def prepare_server_eval_metrics(strategy, server_round):
    """Prepare server evaluation metrics for JSON logging and model cards.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number

    Returns:
        dict: Structured metrics dict for JSON serialization
    """
    # Get the latest eval loss
    composite_eval_loss = (
        strategy.server_eval_losses[-1]
        if hasattr(strategy, "server_eval_losses") and strategy.server_eval_losses
        else "N/A"
    )

    # Get aggregated client metrics
    aggregated_client_metrics = (
        strategy.last_aggregated_metrics
        if hasattr(strategy, "last_aggregated_metrics") and strategy.last_aggregated_metrics
        else {}
    )

    # Get individual client metrics
    individual_client_metrics = (
        strategy.last_client_metrics
        if hasattr(strategy, "last_client_metrics") and strategy.last_client_metrics
        else []
    )

    # Get per-dataset results
    server_eval_dataset_results = (
        strategy.last_per_dataset_results
        if hasattr(strategy, "last_per_dataset_results") and strategy.last_per_dataset_results
        else []
    )

    # Count datasets evaluated
    num_datasets_evaluated = len(server_eval_dataset_results)

    return {
        "composite_eval_loss": composite_eval_loss,
        "aggregated_client_metrics": aggregated_client_metrics,
        "individual_client_metrics": individual_client_metrics,
        "server_eval_dataset_results": server_eval_dataset_results,
        "num_datasets_evaluated": num_datasets_evaluated,
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
