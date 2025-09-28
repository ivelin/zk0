"""zk0: A Flower / Hugging Face LeRobot app."""

from datetime import datetime
from pathlib import Path

from src.task import get_model, get_params
from src.logger import setup_logging
from src.visualization import SmolVLAVisualizer
from loguru import logger

import numpy as np
import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters, FitIns, EvaluateIns, FitRes, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Union, Optional, Dict
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, Scalar, parameters_to_ndarrays

# Import PEFT for LoRA aggregation
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
    logger.warning("PEFT not available, LoRA aggregation disabled")


class AggregateEvaluationStrategy(FedAvg):
    """Custom strategy that aggregates client evaluation results."""

    def __init__(self, *, server_dir: Path = None, log_file: Path = None, save_path: Path = None, evaluate_config_fn=None, num_rounds: int = None, **kwargs):
        # Extract standard FedAvg parameters
        fedavg_kwargs = {k: v for k, v in kwargs.items() if k in FedAvg.__init__.__code__.co_varnames}

        super().__init__(**fedavg_kwargs)
        self.server_dir = server_dir
        self.log_file = log_file
        self.save_path = save_path
        self.evaluate_config_fn = evaluate_config_fn
        self.num_rounds = num_rounds

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure the next round of training."""
        logger.info(f"Server: Configuring fit for round {server_round}")

        # Get base config from parent
        config = super().configure_fit(server_round, parameters, client_manager)
        logger.info(f"Server: Base config generated for {len(config)} clients")

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
            updated_fit_ins = FitIns(
                parameters=fit_ins.parameters,
                config=updated_fit_config
            )
            updated_config.append((client_proxy, updated_fit_ins))

        logger.info(f"Server: Fit configuration complete for round {server_round}")
        return updated_config

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure the evaluation round."""
        logger.info(f"Server: Configuring evaluate for round {server_round}")

        # Get base config from parent
        config = super().configure_evaluate(server_round, parameters, client_manager)
        logger.info(f"Server: Base config generated for {len(config)} clients")

        # Add round number, log_file_path, and save_path for client logging and eval stats
        updated_config = []
        for i, (client_proxy, evaluate_ins) in enumerate(config):
            logger.debug(f"Server: Configuring client {i} (CID: {client_proxy.cid}) for evaluation")
            updated_evaluate_config = evaluate_ins.config.copy()
            updated_evaluate_config["round"] = server_round
            updated_evaluate_config["log_file_path"] = str(self.log_file)
            updated_evaluate_config["save_path"] = str(self.save_path)
            updated_evaluate_config["base_save_path"] = str(self.save_path)
            updated_evaluate_ins = EvaluateIns(
                parameters=evaluate_ins.parameters,
                config=updated_evaluate_config
            )
            updated_config.append((client_proxy, updated_evaluate_ins))

        logger.info(f"Server: Evaluate configuration complete for round {server_round}")
        return updated_config

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""

        logger.info(f"Server: Aggregating evaluation results for round {server_round}")
        logger.info(f"Server: Received {len(results)} successful results, {len(failures)} failures")

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
                visualizer.plot_eval_mse_chart(mse_history, self.server_dir)
                logger.info("Eval MSE chart generated for final round")
            except Exception as e:
                logger.error(f"Failed to generate eval MSE chart: {e}")

        return aggregated_loss, aggregated_metrics


class LoRAFedAvg(FedAvg):
    """FedAvg strategy adapted for LoRA parameter-efficient fine-tuning.

    Aggregates LoRA adapters by averaging A/B matrices separately,
    then merges into the base model. Compatible with Flower's parameter flow.
    """

    def __init__(self, peft_config: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.peft_config = peft_config or {}
        if not self.peft_config.get("enabled", False):
            raise RuntimeError("LoRAFedAvg requires PEFT/LoRA to be enabled")

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate LoRA adapters from clients with enhanced robustness."""
        if not results:
            return None, {}

        logger.info(f"Server: Aggregating LoRA adapters for round {server_round}")

        # Collect adapter parameters from clients using names from metrics
        client_adapters = []
        total_samples = 0
        valid_clients = 0

        for client_proxy, fit_res in results:
            try:
                # Extract parameters from FitRes (list of ndarrays)
                params_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                # Get parameter names from client's training metrics
                param_names = fit_res.metrics.get("lora_param_names", [])
                num_samples = fit_res.num_examples

                # Validate LoRA parameters
                if not param_names:
                    logger.warning(f"Client {client_proxy.cid}: No LoRA parameter names in metrics")
                    continue

                if len(param_names) != len(params_ndarrays):
                    logger.warning(f"Client {client_proxy.cid}: Mismatch between param names ({len(param_names)}) and arrays ({len(params_ndarrays)})")
                    continue

                # Validate that parameters are actually LoRA adapters
                lora_params = [name for name in param_names if "lora" in name.lower()]
                if len(lora_params) != len(param_names):
                    logger.warning(f"Client {client_proxy.cid}: Not all parameters are LoRA ({len(lora_params)}/{len(param_names)})")
                    continue

                # Convert ndarrays to named adapter dict
                adapter_dict = {name: torch.from_numpy(arr) for name, arr in zip(param_names, params_ndarrays)}
                client_adapters.append(adapter_dict)
                total_samples += num_samples
                valid_clients += 1

                logger.debug(f"Client {client_proxy.cid}: Valid LoRA adapters ({len(param_names)} params, {num_samples} samples)")

            except Exception as e:
                logger.warning(f"Client {client_proxy.cid}: Failed to process LoRA adapters: {e}")
                continue

        if not client_adapters:
            logger.error("No valid LoRA adapters received from any clients")
            return None, {}

        logger.info(f"Server: Processing {len(client_adapters)} valid LoRA adapter sets from {valid_clients}/{len(results)} clients")

        # Average adapter parameters by name with robustness checks
        averaged_adapters = {}
        reference_client = client_adapters[0]
        param_names = list(reference_client.keys())

        for param_name in param_names:
            param_tensors = []
            clients_with_param = 0

            for adapter in client_adapters:
                if param_name in adapter:
                    param_tensors.append(adapter[param_name])
                    clients_with_param += 1
                else:
                    logger.warning(f"Parameter {param_name} missing from client adapter")

            if param_tensors:
                # Average available parameters
                averaged_param = torch.mean(torch.stack(param_tensors), dim=0)
                averaged_adapters[param_name] = averaged_param
                logger.debug(f"Averaged {param_name}: {clients_with_param}/{len(client_adapters)} clients contributed")
            else:
                logger.error(f"No clients provided parameter {param_name}")

        if not averaged_adapters:
            logger.error("No parameters could be averaged")
            return None, {}

        logger.info(f"Server: Successfully averaged {len(averaged_adapters)} LoRA adapter parameters from {len(client_adapters)} clients")

        try:
            # Load base model and create PeftModel for merging
            from src.task import get_model, load_data
            trainloader, _ = load_data(0, 4, "smolvla", device="cpu")
            dataset_meta = trainloader.dataset.meta
            base_model = get_model(dataset_meta=dataset_meta, peft_config=self.peft_config)

            # Create PeftModel with averaged adapters
            peft_model = PeftModel.from_pretrained(base_model, averaged_adapters, adapter_name="aggregated")

            # Validate merged model before unloading
            logger.info("Server: Validating merged LoRA model...")
            with torch.no_grad():
                dummy_input = {
                    'image': torch.randn(1, 3, 224, 224),
                    'state': torch.randn(1, 7),
                }
                dummy_task = "test task"
                try:
                    output = peft_model(dummy_input, dummy_task)
                    logger.info("Server: Merged model validation successful")
                except Exception as e:
                    logger.error(f"Server: Merged model validation failed: {e}")
                    raise RuntimeError(f"LoRA merging produced invalid model: {e}")

            # Merge and unload to get final model
            merged_model = peft_model.merge_and_unload()
            logger.info("Server: LoRA adapters successfully merged into base model")

        except Exception as e:
            logger.error(f"Server: Failed to merge LoRA adapters: {e}")
            raise RuntimeError(f"LoRA aggregation failed: {e}")

        # Return merged LoRA parameters as Parameters object (sorted by key for consistency)
        merged_state = merged_model.state_dict()
        lora_keys = sorted([k for k in merged_state.keys() if "lora" in k])
        lora_ndarrays = [merged_state[k].cpu().numpy() for k in lora_keys]
        aggregated_parameters = ndarrays_to_parameters(lora_ndarrays)

        logger.info(f"Server: LoRA aggregation complete - {len(lora_ndarrays)} adapter parameters from {len(lora_keys)} keys")

        return aggregated_parameters, {
            "num_samples": total_samples,
            "lora_keys": lora_keys,
            "valid_clients": valid_clients,
            "total_clients": len(results)
        }


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
            "skip": eval_frequency > 0 and server_round % eval_frequency != 0,
            "eval_mode": eval_mode
        }

    return evaluate_config_fn


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

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

    # Load PEFT config from pyproject.toml [tool.zk0.peft_config]
    try:
        from src.utils import get_tool_config
        peft_config = get_tool_config("zk0.peft_config")
        logger.debug(f"Loaded PEFT config: {peft_config}")
        if not peft_config:
            logger.warning("No PEFT config found in pyproject.toml, using defaults")
            peft_config = {
                "enabled": True,
                "rank": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
                "modules_to_save": ["action_out_proj", "state_proj"]
            }
    except Exception as e:
        logger.warning(f"Failed to load PEFT config from pyproject.toml: {e}, using defaults")
        peft_config = {
            "enabled": True,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["action_out_proj", "state_proj"]
        }

    # Set global model initialization (LoRA adapters for first round)
    # Load a minimal dataset to get metadata for SmolVLA initialization
    from src.task import load_data
    trainloader, _ = load_data(0, 4, "smolvla", device="cpu")  # Use first partition
    dataset_meta = trainloader.dataset.meta
    base_model = get_model(dataset_meta=dataset_meta, peft_config=peft_config)

    # Extract LoRA adapter parameters for initial broadcast
    lora_state = base_model.state_dict()
    lora_ndarrays = [lora_state[k].cpu().numpy() for k in sorted(lora_state.keys()) if "lora" in k]
    logger.info(f"Initial LoRA adapters: {len(lora_ndarrays)} parameters")
    global_model_init = ndarrays_to_parameters(lora_ndarrays)

    # Define strategy with evaluation aggregation
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Add evaluation configuration callback to provide save_path to clients
    eval_frequency = context.run_config.get("eval-frequency", 5)
    eval_mode = context.run_config.get("eval_mode", "quick")
    evaluate_config_fn = get_evaluate_config_callback(save_path, eval_frequency, eval_mode)

    # Use LoRA strategy (LoRA is the only supported mode)
    logger.info("Using LoRAFedAvg strategy for PEFT/LoRA")
    strategy = LoRAFedAvg(
        peft_config=peft_config,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        initial_parameters=global_model_init,
    )

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)