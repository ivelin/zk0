"""zk0: A Flower / Hugging Face LeRobot app."""

from datetime import datetime
from pathlib import Path

from src.task import get_model, get_params
from src.logger import setup_logging
from src.visualization import SmolVLAVisualizer
from loguru import logger

from flwr.common import Context, Metrics, ndarrays_to_parameters, FitIns, EvaluateIns, Parameters, FitRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from typing import List, Tuple, Union, Optional, Dict
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, Scalar

from huggingface_hub import HfApi
from transformers import AutoModel
import torch
from safetensors.torch import save_file

from huggingface_hub import HfApi
from transformers import AutoModel
import torch
from safetensors.torch import save_file


def push_model_to_hub(parameters, server_round: int, models_dir: Path, hf_repo_id: str, wandb_run=None) -> None:
    """Push model checkpoint to Hugging Face Hub.

    Args:
        parameters: Flower Parameters object containing model weights
        server_round: Current server round number
        models_dir: Directory containing the saved checkpoint
        hf_repo_id: Hugging Face repository ID (e.g., "username/repo-name")
        wandb_run: Optional wandb run for logging
    """
    try:
        # Convert Flower Parameters to numpy arrays
        from flwr.common import parameters_to_ndarrays
        ndarrays = parameters_to_ndarrays(parameters)

        # Create a state dict from the numpy arrays
        # We need to create a dummy model to get the parameter names
        from src.task import get_model, load_data
        trainloader, _ = load_data(0, 4, "smolvla", device="cpu")  # Use first partition
        dataset_meta = trainloader.dataset.meta
        model = get_model(dataset_meta)

        # Create state dict with proper parameter names
        state_dict = {}
        for (name, _), ndarray in zip(model.state_dict().items(), ndarrays):
            # Convert numpy array back to torch tensor
            tensor = torch.from_numpy(ndarray)
            # Convert back to the original dtype if it was BFloat16
            original_param = model.state_dict()[name]
            if original_param.dtype == torch.bfloat16:
                tensor = tensor.bfloat16()
            state_dict[name] = tensor

        # Push to Hugging Face Hub
        from huggingface_hub import HfApi
        import os

        # Get HF token from environment
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not found. Required for pushing to Hugging Face Hub.")

        api = HfApi(token=hf_token)

        # Save model locally first, then push
        temp_model_path = models_dir / f"temp_model_round_{server_round}"
        temp_model_path.mkdir(exist_ok=True)

        # Save model config and state dict
        model.config.save_pretrained(temp_model_path)
        torch.save(state_dict, temp_model_path / "pytorch_model.bin")

        # Push to Hub
        api.upload_folder(
            folder_path=str(temp_model_path),
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message=f"Upload federated learning checkpoint from round {server_round}"
        )

        logger.info(f"🚀 Model pushed to Hugging Face Hub: https://huggingface.co/{hf_repo_id}")
        logger.info(f"📊 Pushed {len(state_dict)} parameter tensors to round {server_round}")

        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_model_path)

    except Exception as e:
        logger.error(f"❌ Failed to push model to Hugging Face Hub for round {server_round}: {e}")
        raise


def save_model_checkpoint(parameters, server_round: int, models_dir: Path) -> None:
    """Save model checkpoint to disk using safetensors format.

    Args:
        parameters: Flower Parameters object containing model weights
        server_round: Current server round number
        models_dir: Directory to save the checkpoint
    """
    try:
        # Convert Flower Parameters to numpy arrays
        from flwr.common import parameters_to_ndarrays
        ndarrays = parameters_to_ndarrays(parameters)

        # Create a state dict from the numpy arrays
        # We need to create a dummy model to get the parameter names
        from src.task import get_model, load_data
        trainloader, _ = load_data(0, 4, "smolvla", device="cpu")  # Use first partition
        dataset_meta = trainloader.dataset.meta
        model = get_model(dataset_meta)

        # Create state dict with proper parameter names
        state_dict = {}
        for (name, _), ndarray in zip(model.state_dict().items(), ndarrays):
            # Convert numpy array back to torch tensor
            tensor = torch.from_numpy(ndarray)
            # Convert back to the original dtype if it was BFloat16
            original_param = model.state_dict()[name]
            if original_param.dtype == torch.bfloat16:
                tensor = tensor.bfloat16()
            state_dict[name] = tensor

        # Save using safetensors format
        checkpoint_path = models_dir / f"checkpoint_round_{server_round}.safetensors"
        save_file(state_dict, checkpoint_path)

        logger.info(f"💾 Model checkpoint saved: {checkpoint_path}")
        logger.info(f"📊 Checkpoint contains {len(state_dict)} parameter tensors")

    except Exception as e:
        logger.error(f"❌ Failed to save model checkpoint for round {server_round}: {e}")
        raise


class AggregateEvaluationStrategy(FedProx):
    """Custom strategy that aggregates client evaluation results."""

    def __init__(self, *, proximal_mu: float = 0.01, server_dir: Path = None, models_dir: Path = None, log_file: Path = None, save_path: Path = None, evaluate_config_fn=None, num_rounds: int = None, wandb_run=None, context: Context = None, **kwargs):
        # Store proximal_mu for FedProx
        self.proximal_mu = proximal_mu
        logger.info(f"AggregateEvaluationStrategy: Initialized with proximal_mu={proximal_mu}")
    
        # Extract standard FedProx parameters (including proximal_mu)
        fedprox_kwargs = {k: v for k, v in kwargs.items() if k in FedProx.__init__.__code__.co_varnames}
        # Ensure proximal_mu is included
        fedprox_kwargs['proximal_mu'] = proximal_mu
    
        super().__init__(**fedprox_kwargs)
        self.server_dir = server_dir
        self.models_dir = models_dir
        self.log_file = log_file
        self.save_path = save_path
        self.evaluate_config_fn = evaluate_config_fn
        self.num_rounds = num_rounds
        self.wandb_run = wandb_run
        self.context = context
        self.federated_metrics_history = []  # Track metrics across rounds for plotting

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure the next round of training."""
        logger.info(f"Server: Configuring fit for round {server_round}")

        # Get configuration from pyproject.toml
        from src.utils import get_tool_config
        flwr_config = get_tool_config("flwr", "pyproject.toml")
        app_config = flwr_config.get("app", {}).get("config", {})

        # Get base config from parent
        config = super().configure_fit(server_round, parameters, client_manager)
        logger.info(f"✅ Server: Base config generated for {len(config)} clients")

        # Monitor client availability
        if len(config) == 0:
            logger.error(f"❌ Server: NO CLIENTS AVAILABLE for fit in round {server_round}")
            logger.error("❌ Server: This indicates clients failed to register or crashed during initialization")

        # Log selected clients for this round
        selected_cids = [client_proxy.cid for client_proxy, _ in config]
        logger.info(f"Server: Selected clients for fit in round {server_round}: {selected_cids}")

        # Add round number, log_file_path, and save_path for client logging and eval stats
        updated_config = []
        for i, (client_proxy, fit_ins) in enumerate(config):
            logger.debug(f"Server: Configuring client {i} (CID: {client_proxy.cid})")
            updated_fit_config = fit_ins.config.copy()
            updated_fit_config["round"] = server_round
            updated_fit_config["log_file_path"] = str(self.log_file)
            updated_fit_config["save_path"] = str(self.save_path)
            updated_fit_config["base_save_path"] = str(self.save_path)
            updated_fit_config["timestamp"] = self.save_path.name  # Pass the output folder name to clients
            # FedProx: Add proximal_mu parameter for client-side proximal term calculation
            updated_fit_config["proximal_mu"] = self.proximal_mu
            logger.debug(f"Server: Added proximal_mu={self.proximal_mu} to client {client_proxy.cid} config")

            # Add initial_lr parameter for client-side training
            initial_lr = app_config.get("initial_lr", 1e-3)
            updated_fit_config["initial_lr"] = initial_lr
            logger.debug(f"Server: Added initial_lr={initial_lr} to client {client_proxy.cid} config")

            # + Add use_wandb flag for client-side WandB enablement
            use_wandb = app_config.get("use-wandb", False)
            updated_fit_config["use_wandb"] = use_wandb
            logger.debug(f"Server: Added use_wandb={use_wandb} to client {client_proxy.cid} config")

            updated_fit_ins = FitIns(
                parameters=fit_ins.parameters,
                config=updated_fit_config
            )
            updated_config.append((client_proxy, updated_fit_ins))

        logger.info(f"✅ Server: Fit configuration complete for round {server_round}")
        return updated_config

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure the evaluation round."""
        logger.info(f"Server: Configuring evaluate for round {server_round}")

        # Get base config from parent
        config = super().configure_evaluate(server_round, parameters, client_manager)
        logger.info(f"✅ Server: Base config generated for {len(config)} clients")

        # Monitor client availability for evaluation
        if len(config) == 0:
            logger.error(f"❌ Server: NO CLIENTS AVAILABLE for evaluation in round {server_round}")
            logger.error("❌ Server: This indicates clients failed to register or crashed during initialization")

        # + Load app_config for use-wandb (same as in configure_fit)
        from src.utils import get_tool_config
        flwr_config = get_tool_config("flwr", "pyproject.toml")
        app_config = flwr_config.get("app", {}).get("config", {})

        # Add round number, log_file_path, and save_path for client logging and eval stats
        updated_config = []
        for i, (client_proxy, evaluate_ins) in enumerate(config):
            logger.debug(f"Server: Configuring client {i} (CID: {client_proxy.cid}) for evaluation")
            updated_evaluate_config = evaluate_ins.config.copy()
            updated_evaluate_config["round"] = server_round
            updated_evaluate_config["log_file_path"] = str(self.log_file)
            updated_evaluate_config["save_path"] = str(self.save_path)
            updated_evaluate_config["base_save_path"] = str(self.save_path)

            # + Add use_wandb flag for client-side WandB enablement (even in eval)
            use_wandb = app_config.get("use-wandb", False)
            updated_evaluate_config["use_wandb"] = use_wandb
            logger.debug(f"Server: Added use_wandb={use_wandb} to client {client_proxy.cid} eval config")

            updated_evaluate_ins = EvaluateIns(
                parameters=evaluate_ins.parameters,
                config=updated_evaluate_config
            )
            updated_config.append((client_proxy, updated_evaluate_ins))

        logger.info(f"✅ Server: Evaluate configuration complete for round {server_round}")
        return updated_config

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""

        logger.info(f"Server: Aggregating evaluation results for round {server_round}")
        logger.info(f"📊 Server: Received {len(results)} successful results, {len(failures)} failures")

        # Monitor for excessive failures
        if len(failures) > 0:
            logger.warning(f"⚠️ Server: {len(failures)} client failures in round {server_round}")
            for i, failure in enumerate(failures):
                if isinstance(failure, BaseException):
                    logger.warning(f"  Failure {i}: {type(failure).__name__}: {failure}")
                else:
                    logger.warning(f"  Failure {i}: Client proxy issue")

        if not results:
            logger.warning("Server: No evaluation results received from clients")
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        # Extract custom metrics from client results
        success_rates = []
        action_mses = []
        trajectory_lengths = []

        for _, evaluate_res in results:
            metrics = evaluate_res.metrics
            if "success_rate" in metrics:
                success_rates.append(metrics["success_rate"])
            if "action_mse" in metrics:
                action_mses.append(metrics["action_mse"])
            if "trajectory_length" in metrics:
                trajectory_lengths.append(metrics["trajectory_length"])

        # Aggregate metrics
        aggregated_metrics = {}
        if success_rates:
            aggregated_metrics["avg_success_rate"] = sum(success_rates) / len(success_rates)
        if action_mses:
            aggregated_metrics["avg_action_mse"] = sum(action_mses) / len(action_mses)
        if trajectory_lengths:
            aggregated_metrics["avg_trajectory_length"] = sum(trajectory_lengths) / len(trajectory_lengths)

        # Log aggregated results
        if aggregated_metrics:
            logger.info(f"Round {server_round} aggregated evaluation metrics:")
            for key, value in aggregated_metrics.items():
                logger.info(f"  {key}: {value:.4f}")

        # Track metrics for plotting (add timing and participation info)
        if aggregated_metrics:
            round_metrics = {
                'round': server_round,
                'round_time': 0.0,  # Would need to be tracked separately
                'num_clients': len(results),
                **aggregated_metrics
            }
            self.federated_metrics_history.append(round_metrics)

        # Save aggregated results to file
        if self.server_dir and aggregated_metrics:
            import json
            from datetime import datetime

            server_file = self.server_dir / f"round_{server_round}_aggregated.json"
            data = {
                "round": server_round,
                "timestamp": datetime.now().isoformat(),
                "participation": {
                    "clients": len(results),
                    "failures": len(failures)
                },
                "aggregated_metrics": aggregated_metrics,
                "individual_results": [
                    {
                        "client_id": getattr(client_proxy, 'cid', 'unknown'),
                        "loss": evaluate_res.loss,
                        "metrics": dict(evaluate_res.metrics)
                    }
                    for client_proxy, evaluate_res in results
                ]
            }

            with open(server_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        # Generate chart on the last round
        if self.num_rounds and server_round == self.num_rounds:
            try:
                mse_history = aggregate_eval_mse_history(self.server_dir)
                visualizer = SmolVLAVisualizer()
                visualizer.plot_eval_mse_chart(mse_history, self.server_dir, wandb_run=self.wandb_run)
                # Also plot federated metrics if we have them
                if hasattr(self, 'federated_metrics_history') and self.federated_metrics_history:
                    visualizer.plot_federated_metrics(self.federated_metrics_history, self.server_dir, wandb_run=self.wandb_run)
                logger.info("Eval MSE chart generated for final round")
            except Exception as e:
                logger.error(f"Failed to generate eval MSE chart: {e}")

        return aggregated_loss, aggregated_metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results from clients."""
        logger.info(f"Server: Aggregating fit results for round {server_round}")
        logger.info(f"📊 Server: Received {len(results)} successful results, {len(failures)} failures")

        # Monitor for excessive failures
        if len(failures) > 0:
            logger.warning(f"⚠️ Server: {len(failures)} client failures in fit round {server_round}")
            for i, failure in enumerate(failures):
                if isinstance(failure, BaseException):
                    logger.warning(f"  Failure {i}: {type(failure).__name__}: {failure}")
                else:
                    logger.warning(f"  Failure {i}: Client proxy issue")

        # Call parent aggregate_fit (FedProx) and get aggregated parameters
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        # Log parameter update information
        if aggregated_parameters is not None:
            logger.info(f"✅ Server: Successfully aggregated parameters from {len(results)} clients for round {server_round}")
            # Log parameter norm to track changes
            try:
                import numpy as np
                param_norms = [np.linalg.norm(param) for param in aggregated_parameters.tensors]
                total_norm = sum(param_norms)
                logger.info(f"📊 Server: Aggregated parameters norm: {total_norm:.4f} (from {len(param_norms)} parameter arrays)")
            except Exception as e:
                logger.warning(f"Could not compute parameter norms: {e}")

            # Save model checkpoint based on checkpoint_interval configuration
            checkpoint_interval = self.context.run_config.get("checkpoint_interval", 5)
            if checkpoint_interval > 0 and server_round % checkpoint_interval == 0:
                try:
                    logger.info(f"💾 Server: Saving model checkpoint for round {server_round} (interval: {checkpoint_interval})")
                    save_model_checkpoint(aggregated_parameters, server_round, self.models_dir)
                    logger.info(f"✅ Server: Model checkpoint saved successfully for round {server_round}")
                except Exception as e:
                    logger.error(f"❌ Server: Failed to save model checkpoint for round {server_round}: {e}")

            # Save final model checkpoint at the end of training (regardless of checkpoint_interval)
            if self.num_rounds and server_round == self.num_rounds:
                try:
                    logger.info(f"💾 Server: Saving final model checkpoint for round {server_round} (end of training)")
                    save_model_checkpoint(aggregated_parameters, server_round, self.models_dir)
                    logger.info(f"✅ Server: Final model checkpoint saved successfully for round {server_round}")

                    # Push to Hugging Face Hub if configured
                    hf_repo_id = self.context.run_config.get("hf_repo_id")
                    if hf_repo_id:
                        logger.info(f"🚀 Server: Pushing final model to Hugging Face Hub: {hf_repo_id}")
                        push_model_to_hub(aggregated_parameters, server_round, self.models_dir, hf_repo_id, self.wandb_run)
                        logger.info(f"✅ Server: Model pushed to Hugging Face Hub successfully")
                    else:
                        logger.info("ℹ️ Server: No hf_repo_id configured, skipping Hub push")

                except Exception as e:
                    logger.error(f"❌ Server: Failed to save final model or push to Hub: {e}")
        else:
            logger.warning(f"⚠️ Server: No parameters aggregated for round {server_round}")

        return aggregated_parameters, metrics


def aggregate_eval_mse_history(server_dir: Path) -> Dict[int, Dict[str, float]]:
    """Aggregate evaluation MSE history from server aggregated JSON files.

    Args:
        server_dir: Directory containing round_N_aggregated.json files.

    Returns:
        Dict where keys are round numbers, values are dicts with client MSEs and server avg.

    Raises:
        ValueError: If no evaluation data is found.
    """
    import json
    mse_history = {}

    # Find all server aggregated files
    server_files = list(server_dir.glob("round_*_aggregated.json"))
    if not server_files:
        raise ValueError("No server evaluation data found. Ensure evaluation occurred.")

    for server_file in server_files:
        # Extract round number from filename (round_N_aggregated.json)
        parts = server_file.stem.split('_')
        if len(parts) >= 2 and parts[0] == 'round':
            try:
                round_num = int(parts[1])
            except ValueError:
                continue

            try:
                with open(server_file, 'r') as f:
                    server_data = json.load(f)

                round_data = {}

                # Extract client MSEs from individual_results
                individual_results = server_data.get('individual_results', [])
                for result in individual_results:
                    metrics = result.get('metrics', {})
                    partition_id = metrics.get('partition_id')
                    mse_val = metrics.get('action_mse')
                    if partition_id is not None and mse_val is not None:
                        round_data[f'client_{partition_id}'] = float(mse_val)

                # Extract server average
                aggregated_metrics = server_data.get('aggregated_metrics', {})
                server_avg = aggregated_metrics.get('avg_action_mse')
                if server_avg is not None:
                    round_data['server_avg'] = float(server_avg)

                if round_data:
                    mse_history[round_num] = round_data

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse server aggregated file {server_file}: {e}")
                continue

    if not mse_history:
        raise ValueError("No valid evaluation MSE data found in server files.")

    return mse_history


def get_evaluate_config_callback(save_path: Path, eval_frequency: int = 5, eval_mode: str = "quick"):
    """Return a function to configure an evaluate round.

    Args:
        save_path: Base path for saving evaluation results
        eval_frequency: Evaluate every N rounds (default: 5, 0 = every round)
        eval_mode: Evaluation mode ('quick' or 'full')
    """

    def evaluate_config_fn(server_round: int) -> Metrics:
        eval_save_path = save_path / "evaluate" / f"round_{server_round}"
        return {
            "save_path": str(eval_save_path),
            "base_save_path": str(save_path),
            "round": server_round,
            "eval_mode": eval_mode
        }

    return evaluate_config_fn


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    logger.info(f"🔧 Server: Initializing with {num_rounds} rounds")

    # Create output directory given timestamp (use env var if available, else current time)
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(f"outputs/{folder_name}")
    save_path.mkdir(parents=True, exist_ok=True)

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
    setup_logging(simulation_log_path, client_id="server")
    logger.info("Server logging initialized")

    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.debug("Environment variables loaded from .env file")
    except ImportError:
        logger.debug("python-dotenv not available, skipping .env loading")

    # Get wandb configuration from pyproject.toml
    from src.utils import get_tool_config
    flwr_config = get_tool_config("flwr", "pyproject.toml")
    app_config = flwr_config.get("app", {}).get("config", {})

    # Add app-specific configs to context.run_config for strategy access
    context.run_config["checkpoint_interval"] = app_config.get("checkpoint_interval", 2)

    # Initialize wandb if enabled and API key is available
    wandb_run = None
    if app_config.get("use-wandb", False):
        try:
            import os
            import wandb

            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if wandb_api_key:
                wandb_run = wandb.init(
                    project="zk0",  # + Align with client project for unified dashboard
                    name=f"fl-run-{folder_name}",
                    config=dict(app_config),
                    dir=str(save_path)
                )
                logger.info(f"Wandb initialized: {wandb_run.name} in project {wandb_run.project}")
            else:
                logger.warning("WANDB_API_KEY not found in environment variables. Wandb logging disabled.")
        except ImportError:
            logger.warning("wandb not available. Install with: pip install wandb")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")

    # Store wandb run in context for access by visualization functions
    context.run_config["wandb_run"] = wandb_run

    # Add save_path and log_file_path to run config for clients (for client log paths)
    context.run_config["log_file_path"] = str(simulation_log_path)
    context.run_config["save_path"] = str(save_path)

    # Save configuration snapshot
    import json
    config_snapshot = {
        "timestamp": current_time.isoformat(),
        "run_config": dict(context.run_config),
        "federation": context.run_config.get("federation", "default"),
        "output_structure": {
            "base_dir": str(save_path),
            "simulation_log": str(simulation_log_path),
            "config_file": str(save_path / "config.json"),
            "clients_dir": str(clients_dir),
            "server_dir": str(server_dir),
            "models_dir": str(models_dir)
        }
    }
    with open(save_path / "config.json", 'w') as f:
        json.dump(config_snapshot, f, indent=2, default=str)

    # Set global model initialization
    # Load a minimal dataset to get metadata for SmolVLA initialization
    from src.task import load_data
    trainloader, _ = load_data(0, 4, "smolvla", device="cpu")  # Use first partition
    dataset_meta = trainloader.dataset.meta
    ndarrays = get_params(get_model(dataset_meta=dataset_meta))
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define strategy with evaluation aggregation
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Add evaluation configuration callback to provide save_path to clients
    eval_frequency = context.run_config.get("eval-frequency", 5)
    eval_mode = context.run_config.get("eval_mode", "quick")
    logger.info(f"Server: Using eval_frequency={eval_frequency}, eval_mode={eval_mode}")
    evaluate_config_fn = get_evaluate_config_callback(save_path, eval_frequency, eval_mode)

    # FedProx requires proximal_mu parameter - get from config or use default
    from src.utils import get_tool_config
    flwr_config = get_tool_config("flwr", "pyproject.toml")
    app_config = flwr_config.get("app", {}).get("config", {})
    proximal_mu = app_config.get("proximal_mu", 0.01)
    logger.info(f"Server: Using proximal_mu={proximal_mu} for FedProx strategy")

    strategy = AggregateEvaluationStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        initial_parameters=global_model_init,
        proximal_mu=proximal_mu,  # Required parameter for FedProx
        evaluate_every_round=eval_frequency,  # Respect eval_frequency to skip evaluate calls
        server_dir=server_dir,
        models_dir=models_dir,
        log_file=simulation_log_path,
        save_path=save_path,
        evaluate_config_fn=evaluate_config_fn,
        num_rounds=num_rounds,  # Pass total rounds for chart generation
        wandb_run=wandb_run,  # Pass wandb run for logging
        context=context,  # Pass context for checkpoint configuration
    )

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)