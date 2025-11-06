"""Model saving and Hugging Face Hub utilities for federated learning."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from loguru import logger

from src.common.utils import get_tool_config


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
            create_git_tags(datetime.now().isoformat(), get_project_version(), round_num)
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


def get_project_version():
    """Get project version using importlib.metadata."""
    try:
        from importlib.metadata import version

        return version("zk0")
    except Exception as e:
        logger.warning(f"Failed to get project version: {e}")
        return "unknown"


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
        from src.server.server_utils import prepare_server_eval_metrics
        metrics = prepare_server_eval_metrics(strategy, server_round)
        insights = compute_in_memory_insights(strategy)

        try:
            project_version = get_project_version()
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
        from src.server.server_utils import prepare_server_eval_metrics
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
        f"**Timestamp**: {timestamp}",
        f"**Version**: {version}",
    ]

    content_lines.extend([
        "",
        "### Hyperparameters",
        "",
        f"- **Server Rounds**: {hyperparams.get('num_server_rounds', 'N/A')}",
        f"- **Local Epochs**: {hyperparams.get('local_epochs', 'N/A')}",
        f"- **Proximal μ**: {hyperparams.get('proximal_mu', 'N/A')}",
        f"- **Initial Learning Rate**: {hyperparams.get('initial_lr', 'N/A')}",
        f"- **Batch Size**: {hyperparams.get('batch_size', 'N/A')}",
        f"- **Fraction Fit**: {hyperparams.get('fraction_fit', 'N/A')}",
        f"- **Fraction Evaluate**: {hyperparams.get('fraction_evaluate', 'N/A')}",
        f"- **Eval Frequency**: {hyperparams.get('eval_frequency', 'N/A')}",
        f"- **Eval Batches**: {hyperparams.get('eval_batches', 'N/A')}",
        f"- **Checkpoint Interval**: {hyperparams.get('checkpoint_interval', 'N/A')}",
        f"- **Dynamic Training Decay**: {hyperparams.get('dynamic_training_decay', 'N/A')}",
        f"- **Scheduler Type**: {hyperparams.get('scheduler_type', 'N/A')}",
        f"- **Adaptive LR Enabled**: {hyperparams.get('adaptive_lr_enabled', 'N/A')}",
        f"- **Adaptive μ Enabled**: {hyperparams.get('adaptive_mu_enabled', 'N/A')}",
        "",
        "### Training Datasets",
        "",
    ])

    for ds in train_datasets:
        content_lines.append(f"- **{ds['name']}**: {ds['description']}")

    content_lines.extend([
        "",
        "### Evaluation Datasets",
        "",
    ])

    for ds in eval_datasets:
        content_lines.append(f"- **{ds['name']}**: {ds['description']}")

    content_lines.extend([
        "",
        "### Final Evaluation Metrics",
        "",
        f"- **Composite Eval Loss**: {metrics.get('server_composite_eval_loss', 'N/A')}",
        f"- **Aggregated Client Metrics**: {metrics.get('aggregated_client_metrics', {})}",
        f"- **Individual Client Metrics**: {len(metrics.get('individual_client_metrics', []))} clients",
        "### Per-Dataset Results",
        "",
    ])

    for ds_result in metrics.get("server_eval_dataset_results", []):
        content_lines.append(f"- **{ds_result.get('dataset_name', 'Unknown')}**: Loss {ds_result.get('loss', 'N/A'):.4f}")

    content_lines.extend([
        "",
        "### Training Insights",
        "",
    ])

    # Only include insights that are not N/A
    convergence_trend = insights.get("convergence_trend", "N/A")
    if convergence_trend != "N/A":
        content_lines.append(f"- **Convergence Trend**: {convergence_trend}")

    avg_client_loss_trend = insights.get("avg_client_loss_trend", "N/A")
    if avg_client_loss_trend != "N/A":
        content_lines.append(f"- **Avg Client Loss Trend**: {avg_client_loss_trend}")

    param_update_norm_trend = insights.get("param_update_norm_trend", "N/A")
    if param_update_norm_trend != "N/A":
        content_lines.append(f"- **Param Update Norm Trend**: {param_update_norm_trend}")

    lr_mu_adjustments = insights.get("lr_mu_adjustments", "N/A")
    if lr_mu_adjustments != "N/A":
        content_lines.append(f"- **LR/μ Adjustments**: {lr_mu_adjustments}")

    client_participation_rate = insights.get("client_participation_rate", "N/A")
    if client_participation_rate != "N/A":
        content_lines.append(f"- **Client Participation Rate**: {client_participation_rate}")

    anomalies = insights.get("anomalies", [])
    content_lines.append(f"- **Anomalies**: {', '.join(anomalies) or 'None detected'}")

    content_lines.extend([
        "",
        "## Usage",
        "",
        "This model can be used for robotics manipulation tasks. Load with:",
        "",
        "```python",
        "from lerobot.policies.smolvla import SmolVLAPolicy",
        "",
        f"policy = SmolVLAPolicy.from_pretrained(\"{hf_repo_id or 'your-model-id'}\")",
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