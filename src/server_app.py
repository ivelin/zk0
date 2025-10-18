"""zk0: A Flower / Hugging Face LeRobot app."""

from datetime import datetime
from pathlib import Path

from src.task import get_model, get_params, compute_param_norms, set_params


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

    from flwr.common import parameters_to_ndarrays
    import numpy as np

    current_ndarrays = parameters_to_ndarrays(current_params)
    previous_ndarrays = parameters_to_ndarrays(previous_params)

    if len(current_ndarrays) != len(previous_ndarrays):
        return 0.0

    param_diff_norm = np.sqrt(
        sum(np.sum((c - p) ** 2) for c, p in zip(current_ndarrays, previous_ndarrays))
    )
    return float(param_diff_norm)


def check_early_stopping(
    eval_loss: float, best_loss: float, rounds_without_improvement: int, patience: int
) -> tuple[bool, int]:
    """Check if early stopping should be triggered based on evaluation loss.

    Args:
        eval_loss: Current evaluation loss
        best_loss: Best evaluation loss seen so far
        rounds_without_improvement: Current count of rounds without improvement
        patience: Number of rounds to wait before stopping

    Returns:
        tuple: (should_stop, updated_rounds_without_improvement)
    """
    if patience <= 0:
        return False, 0

    if eval_loss < best_loss:
        # Improvement detected
        return False, 0
    else:
        # No improvement
        new_rounds_without_improvement = rounds_without_improvement + 1
        should_stop = new_rounds_without_improvement >= patience
        return should_stop, new_rounds_without_improvement


def update_early_stopping_tracking(
    strategy, server_round: int, eval_loss: float
) -> None:
    """Update early stopping tracking and log status.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number
        eval_loss: Current evaluation loss
    """
    if strategy.early_stopping_triggered:
        return

    should_stop, strategy.rounds_without_improvement = check_early_stopping(
        eval_loss=eval_loss,
        best_loss=strategy.best_eval_loss,
        rounds_without_improvement=strategy.rounds_without_improvement,
        patience=strategy.early_stopping_patience,
    )

    if eval_loss < strategy.best_eval_loss:
        strategy.best_eval_loss = eval_loss
        logger.info(f"🆕 New best eval loss: {eval_loss:.4f} (round {server_round})")
    else:
        logger.info(
            f"📈 No improvement in eval loss for {strategy.rounds_without_improvement}/{strategy.early_stopping_patience} rounds"
        )

    if should_stop:
        strategy.early_stopping_triggered = True
        logger.warning(
            f"🛑 Early stopping triggered after {server_round} rounds (no eval loss improvement for {strategy.early_stopping_patience} rounds)"
        )
        logger.warning(
            f"   Best eval loss: {strategy.best_eval_loss:.4f}, Current: {eval_loss:.4f}"
        )


def aggregate_client_metrics(validated_results):
    """Aggregate client metrics from validated fit results.

    Args:
        validated_results: List of (client_proxy, fit_result) tuples

    Returns:
        dict: Aggregated client metrics
    """
    return _compute_aggregated_metrics(validated_results)


def _compute_aggregated_metrics(validated_results):
    """Compute aggregated metrics from validated client results.

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
    return _collect_client_metrics(validated_results)


def _collect_client_metrics(validated_results):
    """Collect individual client metrics from validated results.

    Args:
        validated_results: List of (client_proxy, fit_result) tuples

    Returns:
        list: List of individual client metric dictionaries
    """
    from src.utils import create_client_metrics_dict

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


from src.logger import setup_logging
from src.visualization import SmolVLAVisualizer
from src.utils import compute_parameter_hash
from loguru import logger

from flwr.common import (
    Context,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, Scalar

import torch
from safetensors.torch import save_file


class AggregateEvaluationStrategy(FedProx):
    """Custom strategy that aggregates client evaluation results."""

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        proximal_mu: float = 0.01,
        server_dir: Path = None,
        models_dir: Path = None,
        log_file: Path = None,
        save_path: Path = None,
        num_rounds: int = None,
        wandb_run=None,
        context: Context = None,
    ) -> None:
        # Call FedProx super().__init__ with all standard params
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            proximal_mu=proximal_mu,
        )

        # Log CUDA availability on instantiation
        logger.info(f"Server: Instantiated - CUDA available: {torch.cuda.is_available()}")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Server: Using device {self.device}")

        # Custom params
        self.server_dir = server_dir
        self.models_dir = models_dir
        self.log_file = log_file
        self.save_path = save_path
        self.num_rounds = num_rounds
        self.wandb_run = wandb_run
        self.context = context
        self.federated_metrics_history = []  # Track metrics across rounds for plotting
        self.current_parameters = (
            None  # Store current global model parameters for server evaluation
        )
        self.previous_parameters = (
            None  # Store previous round parameters for update norm calculation
        )
        self.last_aggregated_metrics = {}  # Store last round's aggregated client metrics
        self.last_client_metrics = []  # Store last round's individual client metrics
        self.initial_parameters = (
            initial_parameters  # Store initial parameters for early stopping fallback
        )

        # Get eval_frequency from config (default 1)
        self.eval_frequency = (
            context.run_config.get("eval-frequency", 1) if context else 1
        )

        # Early stopping configuration
        self.early_stopping_patience = (
            context.run_config.get("early_stopping_patience", 10) if context else 10
        )
        self.best_eval_loss = float("inf")
        self.rounds_without_improvement = 0
        self.early_stopping_triggered = False

        logger.info(
            f"AggregateEvaluationStrategy: Initialized with proximal_mu={proximal_mu}, eval_frequency={self.eval_frequency}, early_stopping_patience={self.early_stopping_patience}"
        )

        # Override evaluate_fn for server-side evaluation (called by strategy.evaluate every round, gated by frequency)
        # This replaces the default None evaluate_fn to enable server-side eval via Flower's standard flow
        self.evaluate_fn = self._server_evaluate

        # Create reusable model template for parameter name extraction (server is stateful, no race conditions)
        try:
            from src.utils import load_lerobot_dataset
            from src.configs import DatasetConfig
            from src.task import get_model

            dataset_config = DatasetConfig.load()
            server_config = dataset_config.server[0]
            dataset = load_lerobot_dataset(server_config.name)
            dataset_meta = dataset.meta
            self.template_model = get_model(dataset_meta=dataset_meta)
            logger.info(
                "✅ Server: Created reusable model template for parameter operations"
            )
        except Exception as e:
            logger.error(f"❌ Server: Failed to create model template: {e}")
            raise RuntimeError(
                f"Critical error: Cannot create model template for server operations: {e}"
            ) from e

    def _server_evaluate(
        self, server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation function (called by strategy.evaluate). Gated by eval_frequency.

        This replaces client-side evaluation with server-only eval using dedicated datasets.
        Called automatically by Flower's strategy.evaluate after each fit round.
        """
        # Gate by frequency (skip if not time for eval) - prevents unnecessary evaluations
        if server_round % self.eval_frequency != 0:
            logger.info(
                f"ℹ️ Server: Skipping _server_evaluate for round {server_round} (not multiple of eval_frequency={self.eval_frequency})"
            )
            return None

        # Log CUDA before evaluation
        logger.info(
            f"Server: Starting evaluation round {server_round} - CUDA available: {torch.cuda.is_available()}"
        )

        logger.info(
            f"🔍 Server: _server_evaluate called for round {server_round} (frequency check passed)"
        )

        try:
            from src.task import test, get_model, set_params
            from src.utils import load_lerobot_dataset
            from src.configs import DatasetConfig
            from flwr.common import ndarrays_to_parameters

            # Store parameters for use in aggregate_fit if needed
            self.current_parameters = ndarrays_to_parameters(parameters)

            logger.info(f"🔍 Server: Loading DatasetConfig...")
            dataset_config = DatasetConfig.load()
            logger.info(
                f"🔍 Server: config.server length: {len(dataset_config.server) if dataset_config.server else 0}"
            )
            if dataset_config.server:
                logger.info(
                    f"🔍 Server: First server dataset: {dataset_config.server[0].name}"
                )

            if not dataset_config.server:
                raise ValueError("No server evaluation dataset configured")

            server_config = dataset_config.server[0]
            logger.info(f"🔍 Server: Loading dataset '{server_config.name}'...")
            dataset = load_lerobot_dataset(server_config.name)
            logger.info(
                f"✅ Server: Dataset loaded successfully (episodes: {len(dataset) if hasattr(dataset, '__len__') else 'unknown'})"
            )
            dataset_meta = dataset.meta
            logger.info(
                f"🔍 Server: dataset_meta info keys: {list(dataset_meta.info.keys()) if dataset_meta else 'None'}"
            )

            # Use cached template model for evaluation (no redundant creation)
            logger.info(f"🔍 Server: Using cached template model for evaluation...")
            model = self.template_model
            logger.info(
                f"✅ Server: Template model ready (total params: {sum(p.numel() for p in model.parameters())}"
            )

            # Set parameters
            logger.info(f"🔍 Server: Setting parameters...")
            set_params(model, parameters)
            logger.info(f"✅ Server: Parameters set successfully")

            # Move model to device
            model = model.to(self.device)
            logger.info(f"✅ Server: Model moved to device '{self.device}'")

            # Perform evaluation
            eval_batches = self.context.run_config.get("eval_batches", 0)
            logger.info(
                f"🔍 Server: Running test() on device '{self.device}' with eval_batches={eval_batches}"
            )
            loss, num_examples, metrics = test(
                model, device=self.device, eval_batches=eval_batches
            )
            logger.info(
                f"✅ Server: test() completed - loss={loss}, num_examples={num_examples}, metrics keys={list(metrics.keys()) if metrics else 'Empty'}"
            )
            logger.info(
                f"Server evaluation round {server_round}: loss={loss:.4f}, num_examples={num_examples}"
            )
            logger.info(f"Server evaluation metrics: {metrics}")

            # Log to WandB
            if self.wandb_run:
                from src.wandb_utils import log_wandb_metrics
                from src.utils import prepare_server_wandb_metrics

                # Use utility function to prepare WandB metrics with same structure as JSON files
                # This ensures WandB metrics structure matches JSON file structure
                wandb_metrics = prepare_server_wandb_metrics(
                    server_round=server_round,
                    server_loss=loss,
                    server_metrics=metrics,
                    aggregated_client_metrics=self.last_aggregated_metrics,
                    individual_client_metrics=self.last_client_metrics,
                )

                log_wandb_metrics(wandb_metrics, step=server_round)
                logger.debug(
                    f"Logged server eval + client metrics to WandB using utility function: {list(wandb_metrics.keys())}"
                )

            # Track metrics for plotting
            round_metrics = {
                "round": server_round,
                "round_time": 0.0,
                "num_clients": self.last_aggregated_metrics.get("num_clients", 0),
                "avg_policy_loss": metrics.get("policy_loss", 0.0),
                "avg_client_loss": self.last_aggregated_metrics.get(
                    "avg_client_loss", 0.0
                ),
                "param_update_norm": self.last_aggregated_metrics.get(
                    "param_update_norm", 0.0
                ),
            }
            self.federated_metrics_history.append(round_metrics)

            # Update early stopping tracking
            update_early_stopping_tracking(self, server_round, loss)

            # Track server eval losses for dynamic adjustment
            if not hasattr(self, "server_eval_losses"):
                self.server_eval_losses = []
            self.server_eval_losses.append(loss)
            # Keep only last 10 losses to prevent unbounded growth
            if len(self.server_eval_losses) > 10:
                self.server_eval_losses = self.server_eval_losses[-10:]

            # Save evaluation results to file
            if self.server_dir:
                import json
                from datetime import datetime

                # Fix metrics bug: Update round number in individual_client_metrics before saving
                for metric in self.last_client_metrics:
                    metric["round"] = server_round

                server_file = self.server_dir / f"round_{server_round}_server_eval.json"
                data = {
                    "round": server_round,
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_type": "server_side",
                    "loss": loss,
                    "num_examples": num_examples,
                    "metrics": metrics,
                    "aggregated_client_metrics": self.last_aggregated_metrics,  # Consolidated aggregated metrics
                    "individual_client_metrics": self.last_client_metrics,  # Individual client metrics with IDs
                    "metrics_descriptions": {
                        "policy_loss": "Average policy forward loss per batch (primary evaluation metric for SmolVLA flow-matching model)",
                        "action_dim": "Number of action dimensions detected from batch (default 7 for SO-100 joints + gripper)",
                        "successful_batches": "Number of batches successfully processed during evaluation",
                        "total_batches_processed": "Total batches attempted (including failed)",
                        "total_samples": "Total number of action samples evaluated",
                        "aggregated_client_metrics": {
                            "avg_client_loss": "Average total training loss (policy + fedprox) across all clients in this round",
                            "std_client_loss": "Standard deviation of client total training losses",
                            "avg_client_proximal_loss": "Average FedProx proximal regularization loss across clients",
                            "avg_client_grad_norm": "Average gradient norm across clients",
                            "num_clients": "Number of clients that participated in this round",
                            "param_update_norm": "L2 norm of parameter changes from previous round",
                        },
                        "individual_client_metrics": {
                            "client_id": "Unique client identifier (corresponds to dataset partition)",
                            "loss": "Total training loss for this client (policy_loss + fedprox_loss)",
                            "policy_loss": "Pure SmolVLA flow-matching training loss for this client",
                            "fedprox_loss": "FedProx proximal regularization loss for this client (added to policy_loss during training: total_loss = policy_loss + fedprox_loss)",
                            "grad_norm": "Gradient norm for this client",
                            "param_hash": "SHA256 hash of client's parameter update",
                            "dataset_name": "Name of the dataset this client is training on",
                            "num_steps": "Number of training steps completed by this client",
                            "param_update_norm": "L2 norm of parameter changes from global model",
                            "flower_proxy_cid": "Flower internal client proxy identifier (for debugging)",
                            "round": "The server round this metric was collected in",
                        },
                    },
                }

                with open(server_file, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                logger.info(f"✅ Server: Eval results saved to {server_file}")

            # Generate chart on last round
            if self.num_rounds and server_round == self.num_rounds:
                try:
                    from src.visualization import SmolVLAVisualizer

                    policy_loss_history = aggregate_eval_policy_loss_history(
                        self.server_dir
                    )
                    visualizer = SmolVLAVisualizer()
                    visualizer.plot_eval_policy_loss_chart(
                        policy_loss_history, self.server_dir, wandb_run=self.wandb_run
                    )
                    if self.federated_metrics_history:
                        visualizer.plot_federated_metrics(
                            self.federated_metrics_history,
                            self.server_dir,
                            wandb_run=self.wandb_run,
                        )

                    from src.wandb_utils import finish_wandb

                    finish_wandb()
                    logger.info("WandB run finished after final round")

                    logger.info("Eval MSE chart generated for final round")
                except Exception as e:
                    logger.error(f"Failed to generate eval MSE chart: {e}")

            logger.info(
                f"✅ Server: _server_evaluate completed for round {server_round}"
            )
            return loss, metrics

        except Exception as e:
            logger.error(
                f"❌ Server: Failed _server_evaluate for round {server_round}: {e}"
            )
            logger.error(f"🔍 Detailed error: type={type(e).__name__}, args={e.args}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None, {}

    def compute_fedprox_parameters(self, server_round: int, app_config: Dict[str, Scalar]) -> Tuple[float, float]:
        """Compute FedProx mu and learning rate parameters for the current round.

        Args:
            server_round: Current server round number
            app_config: Application configuration from pyproject.toml

        Returns:
            tuple: (current_mu, current_lr) for this round
        """
        # FedProx: Dynamically adjust proximal_mu and LR based on evaluation trends
        initial_mu = self.proximal_mu
        current_mu = initial_mu
        # Track current LR across rounds (initialize if not set)
        if not hasattr(self, "current_lr"):
            self.current_lr = app_config.get("initial_lr", 1e-3)
        current_lr = self.current_lr

        # Check if dynamic training decay is enabled
        dynamic_training_decay = app_config.get("dynamic_training_decay", False)
        if dynamic_training_decay and hasattr(self, "server_eval_losses") and len(self.server_eval_losses) >= 3:
            from src.task import compute_joint_adjustment
            current_mu, current_lr, reason = compute_joint_adjustment(
                self.server_eval_losses, initial_mu, current_lr
            )
            logger.info(
                f"Server R{server_round}: Dynamic decay mu={current_mu:.6f}, lr={current_lr:.6f} ({reason}, eval_trend={self.server_eval_losses[-3:]})"
            )
            # Update tracked LR for next round
            self.current_lr = current_lr
        else:
            # Fallback to fixed halving for early rounds or when disabled
            current_mu = initial_mu / (2 ** (server_round // 10))
            logger.info(
                f"Server R{server_round}: Fixed adjust mu={current_mu:.6f} (initial={initial_mu}, factor=2^(server_round//10))"
            )

        return current_mu, current_lr

    def validate_client_parameters(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Validate client parameter hashes and return only validated results.

        Args:
            results: List of (client_proxy, fit_result) tuples from successful clients

        Returns:
            List of validated (client_proxy, fit_result) tuples, excluding corrupted clients
        """
        validated_results = []
        for client_proxy, fit_res in results:
            client_metrics = fit_res.metrics
            if "param_hash" in client_metrics:
                client_hash = client_metrics["param_hash"]

                # Compute hash of client's actual parameters on server side
                from flwr.common import parameters_to_ndarrays

                client_params = parameters_to_ndarrays(fit_res.parameters)
                server_computed_hash = compute_parameter_hash(client_params)

                # Compare hashes
                if server_computed_hash == client_hash:
                    logger.info(
                        f"✅ Server: Client {client_proxy.cid} parameter hash VALIDATED: {client_hash[:8]}..."
                    )
                    validated_results.append((client_proxy, fit_res))
                else:
                    error_msg = f"Parameter hash MISMATCH for client {client_proxy.cid}! Client: {client_hash[:8]}..., Server: {server_computed_hash[:8]}..."
                    logger.error(
                        f"❌ Server: {error_msg} - Excluding corrupted client from aggregation"
                    )
            else:
                # No hash provided, include but log warning
                logger.warning(
                    f"⚠️ Server: Client {client_proxy.cid} provided no parameter hash - including in aggregation"
                )
                validated_results.append((client_proxy, fit_res))

        return validated_results

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure the next round of training."""
        # Log CUDA before training round
        logger.info(
            f"Server: Starting training round {server_round} - CUDA available: {torch.cuda.is_available()}"
        )

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
            logger.error(
                f"❌ Server: NO CLIENTS AVAILABLE for fit in round {server_round}"
            )
            logger.error(
                "❌ Server: This indicates clients failed to register or crashed during initialization"
            )

        # Log selected clients for this round
        selected_cids = [client_proxy.cid for client_proxy, _ in config]
        logger.info(
            f"Server: Selected clients for fit in round {server_round}: {selected_cids}"
        )

        # Add round number, log_file_path, and save_path for client logging and eval stats
        updated_config = []
        for i, (client_proxy, fit_ins) in enumerate(config):
            logger.debug(f"Server: Configuring client {i} (CID: {client_proxy.cid})")
            updated_fit_config = fit_ins.config.copy()
            updated_fit_config["round"] = server_round
            updated_fit_config["log_file_path"] = str(self.log_file)
            updated_fit_config["save_path"] = str(self.save_path)
            updated_fit_config["base_save_path"] = str(self.save_path)
            updated_fit_config["timestamp"] = (
                self.save_path.name
            )  # Pass the output folder name to clients for JSON saving
            # FedProx: Dynamically adjust proximal_mu and LR based on evaluation trends
            current_mu, current_lr = self.compute_fedprox_parameters(server_round, app_config)

            updated_fit_config["proximal_mu"] = current_mu
            updated_fit_config["initial_lr"] = current_lr

            # Add initial_lr parameter for client-side training
            initial_lr = app_config.get("initial_lr", 1e-3)
            updated_fit_config["initial_lr"] = initial_lr
            logger.debug(
                f"Server: Added initial_lr={initial_lr} to client {client_proxy.cid} config"
            )

            # + Add use_wandb flag for client-side WandB enablement
            use_wandb = app_config.get("use-wandb", False)
            updated_fit_config["use_wandb"] = use_wandb
            logger.debug(
                f"Server: Added use_wandb={use_wandb} to client {client_proxy.cid} config"
            )

            # Add batch_size from pyproject.toml to override client defaults
            batch_size = app_config.get("batch_size", 64)
            updated_fit_config["batch_size"] = batch_size
            logger.debug(
                f"Server: Added batch_size={batch_size} to client {client_proxy.cid} config"
            )

            # WandB run_id not needed in fit config - client already initialized wandb in client_fn

            # 🛡️ VALIDATE: Server outgoing parameters (for training) - with detailed logging
            from src.utils import validate_and_log_parameters
            from flwr.common import parameters_to_ndarrays

            fit_ndarrays = parameters_to_ndarrays(fit_ins.parameters)
            logger.debug(
                f"Server: Pre-serialization params for client {client_proxy.cid}: {len(fit_ndarrays)} arrays"
            )
            for j, ndarray in enumerate(fit_ndarrays[:3]):  # Log first 3
                logger.debug(
                    f"  Pre-serial param {j}: shape={ndarray.shape}, dtype={ndarray.dtype}, min={ndarray.min():.4f}, max={ndarray.max():.4f}"
                )
            if len(fit_ndarrays) > 3:
                logger.debug(f"  ... and {len(fit_ndarrays) - 3} more")

            fit_param_hash = validate_and_log_parameters(
                fit_ndarrays, f"server_fit_r{server_round}_client{i}"
            )
            logger.debug(
                f"Server: Computed hash on pre-Flower params: {fit_param_hash}"
            )

            # 🔐 ADD: Include parameter hash in client config for validation
            updated_fit_config["param_hash"] = fit_param_hash

            updated_fit_ins = FitIns(
                parameters=fit_ins.parameters, config=updated_fit_config
            )
            updated_config.append((client_proxy, updated_fit_ins))

        logger.info(f"✅ Server: Fit configuration complete for round {server_round}")
        return updated_config

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure the evaluation round - skip client evaluation (server-side only via evaluate_fn)."""
        logger.info(
            f"ℹ️ Server: configure_evaluate for round {server_round} - skipping client eval (server-side via evaluate_fn)"
        )
        return []  # No client evaluation

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate client evaluation results (policy_loss only)."""
        import numpy as np

        logger.info(f"Server: Aggregating evaluate results for round {server_round}")
        if results:
            # Extract policy_loss from client results
            client_policy_losses = [
                res.metrics.get("policy_loss", 0.0) for _, res in results
            ]
            avg_policy_loss = (
                float(np.mean(client_policy_losses)) if client_policy_losses else 0.0
            )
            logger.info(
                f"Server: Aggregated client policy_loss: {avg_policy_loss:.4f} from {len(results)} clients"
            )
            return avg_policy_loss, {"avg_client_policy_loss": avg_policy_loss}
        else:
            logger.info(
                f"ℹ️ Server: No client evaluation results for round {server_round}"
            )
            return None, {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results from clients."""
        logger.info(f"Server: Aggregating fit results for round {server_round}")
        logger.info(
            f"📊 Server: Received {len(results)} successful results, {len(failures)} failures"
        )

        # Monitor for excessive failures
        if len(failures) > 0:
            logger.warning(
                f"⚠️ Server: {len(failures)} client failures in fit round {server_round}"
            )
            for i, failure in enumerate(failures):
                if isinstance(failure, BaseException):
                    logger.warning(
                        f"  Failure {i}: {type(failure).__name__}: {failure}"
                    )
                else:
                    logger.warning(f"  Failure {i}: Client proxy issue")

        # 🔐 VALIDATE: Individual client parameter hashes BEFORE aggregation
        validated_results = self.validate_client_parameters(results)

        # Aggregate client metrics before calling parent
        aggregated_client_metrics = aggregate_client_metrics(validated_results)
        logger.info(
            f"DEBUG SERVER AGGREGATE: Aggregated metrics: {aggregated_client_metrics}"
        )

        # Call parent aggregate_fit (FedProx) with validated results only
        aggregated_parameters, parent_metrics = super().aggregate_fit(
            server_round, validated_results, failures
        )

        # Compute parameter update norm if we have previous parameters
        if self.previous_parameters is not None and aggregated_parameters is not None:
            param_update_norm = compute_server_param_update_norm(
                self.previous_parameters, aggregated_parameters
            )
            aggregated_client_metrics["param_update_norm"] = param_update_norm

        # Store for use in _server_evaluate
        self.last_aggregated_metrics = aggregated_client_metrics

        # Store individual client metrics for detailed reporting
        self.last_client_metrics = collect_individual_client_metrics(validated_results)

        # Initialize last_client_metrics if not set (for round 0 evaluation)
        if not hasattr(self, "last_client_metrics") or self.last_client_metrics is None:
            self.last_client_metrics = []


        # Merge client metrics with parent metrics
        metrics = {**parent_metrics, **aggregated_client_metrics}

        # DIAGNOSIS METRICS: Add current mu, LR, and eval trend to metrics for JSON/WandB logging
        current_mu = (
            self.proximal_mu
        )  # Initial mu; actual per-round mu adjusted in configure_fit
        current_lr = self.context.run_config.get("initial_lr", "N/A")
        eval_trend = (
            self.server_eval_losses[-3:]
            if hasattr(self, "server_eval_losses") and self.server_eval_losses
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

        # Store the aggregated parameters for server-side evaluation
        self.current_parameters = aggregated_parameters

        # Store previous parameters for next round's update norm calculation
        self.previous_parameters = aggregated_parameters

        # Check if early stopping should terminate training
        if self.early_stopping_triggered:
            logger.warning(
                f"🛑 Early stopping: Terminating training after round {server_round}"
            )
            logger.warning(f"   Best eval loss achieved: {self.best_eval_loss:.4f}")
            logger.warning(
                f"   Rounds without improvement: {self.rounds_without_improvement}"
            )
            # Return the aggregated parameters from this round (or initial if none aggregated)
            # This ensures we always return valid parameters to avoid Flower unpacking errors
            final_parameters = (
                aggregated_parameters
                if aggregated_parameters is not None
                else self.initial_parameters
            )
            logger.info(
                f"✅ Server: Early stopping - returning parameters from round {server_round}"
            )
            return final_parameters, metrics

        # Log post-aggregation global norms (now aggregated_parameters is defined)
        if aggregated_parameters is not None:
            # Import here to avoid scope/shadowing issues
            from flwr.common import parameters_to_ndarrays
            from src.task import set_params

            # Convert Flower Parameters to numpy arrays for set_params
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            # Use cached template model for norm computation (no redundant creation)
            set_params(self.template_model, aggregated_ndarrays)
            post_agg_full_norm, post_agg_full_num, _ = compute_param_norms(
                self.template_model, trainable_only=False
            )
            post_agg_train_norm, post_agg_train_num, _ = compute_param_norms(
                self.template_model, trainable_only=True
            )
            total_params = sum(p.numel() for p in self.template_model.parameters())
            trainable_params = sum(
                p.numel() for p in self.template_model.parameters() if p.requires_grad
            )
            logger.info(
                f"Server R{server_round} POST-AGG: Full norm={post_agg_full_norm:.4f} ({post_agg_full_num} tensors, {total_params} elems), Trainable norm={post_agg_train_norm:.4f} ({post_agg_train_num} tensors, {trainable_params} elems)"
            )

        # Log parameter update information
        if aggregated_parameters is not None:
            logger.info(
                f"✅ Server: Successfully aggregated parameters from {len(results)} clients for round {server_round}"
            )

            # 🛡️ VALIDATE: Server aggregated parameters (import inside to avoid shadowing)
            from flwr.common import parameters_to_ndarrays

            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            from src.utils import validate_and_log_parameters

            aggregated_hash = validate_and_log_parameters(
                aggregated_ndarrays, f"server_aggregated_r{server_round}"
            )

            # 🛡️ VALIDATE: Server incoming parameters (aggregated from validated clients)
            from src.utils import validate_and_log_parameters
            from flwr.common import parameters_to_ndarrays

            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            aggregated_hash = validate_and_log_parameters(
                aggregated_ndarrays, f"server_aggregated_r{server_round}"
            )

            # Save model checkpoint based on checkpoint_interval configuration
            checkpoint_interval = self.context.run_config.get("checkpoint_interval", 5)
            if checkpoint_interval > 0 and server_round % checkpoint_interval == 0:
                try:
                    logger.info(
                        f"💾 Server: Saving model checkpoint for round {server_round} (interval: {checkpoint_interval})"
                    )
                    self.save_model_checkpoint(
                        aggregated_parameters, server_round, self.models_dir
                    )
                    logger.info(
                        f"✅ Server: Model checkpoint saved successfully for round {server_round}"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Server: Failed to save model checkpoint for round {server_round}: {e}"
                    )

            # Save final model checkpoint at the end of training (regardless of checkpoint_interval)
            if self.num_rounds and server_round == self.num_rounds:
                try:
                    logger.info(
                        f"💾 Server: Saving final model checkpoint for round {server_round} (end of training)"
                    )
                    self.save_model_checkpoint(
                        aggregated_parameters, server_round, self.models_dir
                    )
                    logger.info(
                        f"✅ Server: Final model checkpoint saved successfully for round {server_round}"
                    )

                    # Perform final evaluation for the last round
                    try:
                        logger.info(
                            f"🔍 Server: Performing final evaluation for round {server_round}"
                        )
                        # Convert Parameters to NDArrays for _server_evaluate
                        from flwr.common import parameters_to_ndarrays

                        aggregated_ndarrays = parameters_to_ndarrays(
                            aggregated_parameters
                        )
                        self._server_evaluate(server_round, aggregated_ndarrays, {})
                    except Exception as e:
                        logger.error(
                            f"❌ Server: Failed final evaluation for round {server_round}: {e}"
                        )

                    # Push to Hugging Face Hub if configured
                    hf_repo_id = self.context.run_config.get("hf_repo_id")
                    if hf_repo_id:
                        logger.info(
                            f"🚀 Server: Pushing final model to Hugging Face Hub: {hf_repo_id}"
                        )
                        self.push_model_to_hub(
                            aggregated_parameters, server_round, hf_repo_id
                        )
                        logger.info(
                            "✅ Server: Model pushed to Hugging Face Hub successfully"
                        )
                    else:
                        logger.info(
                            "ℹ️ Server: No hf_repo_id configured, skipping Hub push"
                        )

                except Exception as e:
                    logger.error(
                        f"❌ Server: Failed to save final model or push to Hub: {e}"
                    )
        else:
            logger.warning(
                f"⚠️ Server: No parameters aggregated for round {server_round}"
            )

        # CRITICAL: Always return valid parameters tuple to prevent Flower unpacking errors
        # This fixes the "cannot unpack non-iterable NoneType object" error
        if aggregated_parameters is None:
            logger.warning(
                f"⚠️ Server: No parameters aggregated for round {server_round}, returning initial parameters"
            )
            aggregated_parameters = self.initial_parameters

        return aggregated_parameters, metrics


    def save_model_checkpoint(
        self, parameters, server_round: int, models_dir: Path
    ) -> None:
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
            # Use the reusable template model for parameter names
            model = self.template_model

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
            checkpoint_path = (
                models_dir / f"checkpoint_round_{server_round}.safetensors"
            )
            save_file(state_dict, checkpoint_path)

            logger.info(f"💾 Model checkpoint saved: {checkpoint_path}")
            logger.info(f"📊 Checkpoint contains {len(state_dict)} parameter tensors")

        except Exception as e:
            logger.error(
                f"❌ Failed to save model checkpoint for round {server_round}: {e}"
            )
            raise

    def push_model_to_hub(self, parameters, server_round: int, hf_repo_id: str) -> None:
        """Push model checkpoint to Hugging Face Hub.

        Args:
            parameters: Flower Parameters object containing model weights
            server_round: Current server round number
            hf_repo_id: Hugging Face repository ID (e.g., "username/repo-name")
        """
        try:
            # Convert Flower Parameters to numpy arrays
            from flwr.common import parameters_to_ndarrays

            ndarrays = parameters_to_ndarrays(parameters)

            # Create a state dict from the numpy arrays
            # Use the reusable template model for parameter names
            model = self.template_model

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
            logger.info(
                f"🔍 HF_TOKEN check: {'Set' if hf_token else 'MISSING (this will cause 403)'}"
            )
            if not hf_token:
                raise ValueError(
                    "HF_TOKEN environment variable not found. Required for pushing to Hugging Face Hub."
                )

            api = HfApi(token=hf_token)

            # 🔍 Create repo if it doesn't exist (fixes 403 for non-existent repo)
            # This auto-creates the repo to avoid "Cannot access content" errors
            try:
                logger.info(f"🔍 Creating/ensuring repo '{hf_repo_id}' exists...")
                api.create_repo(
                    repo_id=hf_repo_id,
                    repo_type="model",
                    exist_ok=True,
                    private=False,  # Set to True if private repo needed
                )
                logger.info(f"✅ Repo '{hf_repo_id}' created or already exists")
            except Exception as create_err:
                logger.error(f"❌ Repo creation failed: {create_err}")
                raise

            # 🔍 Validate repo existence AFTER creation
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

            # Save model locally first, then push
            temp_model_path = self.models_dir / f"temp_model_round_{server_round}"
            temp_model_path.mkdir(exist_ok=True)

            # Save model config and state dict
            model.config.save_pretrained(temp_model_path)
            torch.save(state_dict, temp_model_path / "pytorch_model.bin")

            logger.info(
                f"🔍 Preparing upload: folder={temp_model_path}, repo={hf_repo_id}, commit='Upload federated learning checkpoint from round {server_round}'"
            )

            # Push to Hub
            api.upload_folder(
                folder_path=str(temp_model_path),
                repo_id=hf_repo_id,
                repo_type="model",
                commit_message=f"Upload federated learning checkpoint from round {server_round}",
            )

            logger.info(
                f"🚀 Model pushed to Hugging Face Hub: https://huggingface.co/{hf_repo_id}"
            )
            logger.info(
                f"📊 Pushed {len(state_dict)} parameter tensors to round {server_round}"
            )

            # Clean up temp directory
            import shutil

            shutil.rmtree(temp_model_path)

        except Exception as e:
            logger.error(
                f"❌ Failed to push model to Hugging Face Hub for round {server_round}: {e}"
            )
            logger.error(f"🔍 Detailed error type: {type(e).__name__}, args: {e.args}")
            raise


def compute_dynamic_mu(client_metrics, cfg):
    """Compute dynamic FedProx mu based on client loss standard deviation.

    Args:
        client_metrics: List of client metric dictionaries
        cfg: Configuration object with mu parameters

    Returns:
        float: Adjusted mu value
    """
    if not cfg.get("adaptive_mu_enabled", False) or len(client_metrics) < 2:
        return cfg.get("proximal_mu", 0.01)

    import numpy as np
    losses = [m["loss"] for m in client_metrics]
    loss_std = np.std(losses)
    threshold = cfg.get("loss_std_threshold", 1.2)

    if loss_std > threshold:
        return cfg["proximal_mu"] * cfg.get("mu_adjust_factor", 1.05)
    return cfg["proximal_mu"]


def adjust_global_lr_for_next_round(server_loss_history, current_lr, cfg):
    """Adjust global learning rate based on server evaluation loss trends.

    Args:
        server_loss_history: List of recent server evaluation losses
        current_lr: Current learning rate
        cfg: Configuration object with LR adjustment parameters

    Returns:
        float: Adjusted learning rate
    """
    if len(server_loss_history) < cfg.get("adjustment_window", 5):
        return current_lr

    recent_losses = server_loss_history[-cfg["adjustment_window"]:]
    improvement = (recent_losses[0] - recent_losses[-1]) / max(recent_losses[0], 1e-8)

    if improvement < 0.01:  # Stall
        factor = 0.95
    elif improvement < -cfg.get("spike_threshold", 0.5):  # Divergence
        factor = 1.05
    else:
        factor = 1.0

    new_lr = current_lr * factor
    return max(new_lr, cfg.get("eta_min", 5e-7))


def is_spike_risk(loss_history, cfg):
    """Check if current loss trends indicate spike risk.

    Args:
        loss_history: List of recent loss values
        cfg: Configuration object with spike detection parameters

    Returns:
        bool: True if spike risk detected
    """
    if len(loss_history) < 3:
        return False

    recent = loss_history[-3:]
    delta = recent[-1] - recent[0]
    return delta > cfg.get("spike_threshold", 0.5)


def prepare_client_context(next_mu, next_lr, client_history):
    """Prepare context dictionary for client configuration.

    Args:
        next_mu: Next round's FedProx mu value
        next_lr: Next round's learning rate
        client_history: Client training history for adaptive boosts

    Returns:
        dict: Context dictionary for Flower client configuration
    """
    return {"next_mu": next_mu, "next_lr": next_lr, "client_history": client_history}


def aggregate_eval_policy_loss_history(server_dir: Path) -> Dict[int, Dict[str, float]]:
    """Aggregate evaluation policy loss history from server eval JSON files.

    Args:
        server_dir: Directory containing round_X_server_eval.json files.

    Returns:
        Dict where keys are round numbers, values are dicts with server policy loss values.

    Raises:
        ValueError: If no evaluation data is found.
    """
    import json

    policy_loss_history = {}

    # Find all server eval files (server-side evaluation)
    server_files = list(server_dir.glob("round_*_server_eval.json"))
    if not server_files:
        raise ValueError(
            "No server evaluation data found. Ensure server-side evaluation occurred."
        )

    for server_file in server_files:
        # Extract round number from filename (round_X_server_eval.json)
        parts = server_file.stem.split("_")
        if len(parts) >= 3 and parts[0] == "round" and parts[2] == "server":
            try:
                round_num = int(parts[1])
            except ValueError:
                continue

            try:
                with open(server_file, "r") as f:
                    server_data = json.load(f)

                round_data = {}

                # Extract server policy loss from metrics
                metrics = server_data.get("metrics", {})
                policy_loss = metrics.get("policy_loss")
                if policy_loss is not None:
                    round_data["server_policy_loss"] = float(policy_loss)
                round_data["action_dim"] = metrics.get("action_dim", 7)

                if round_data:
                    policy_loss_history[round_num] = round_data

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse server eval file {server_file}: {e}")
                continue

    if not policy_loss_history:
        raise ValueError(
            "No valid server evaluation policy loss data found in server files."
        )

    return policy_loss_history


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

    # Initialize WandB if enabled
    from src.wandb_utils import init_server_wandb

    wandb_run = None
    run_id = f"zk0-sim-fl-run-{folder_name}"
    if app_config.get("use-wandb", False):
        wandb_run = init_server_wandb(
            project="zk0",
            run_id=run_id,
            config=dict(app_config),
            dir=str(save_path),
            notes=f"Federated Learning Server - {num_rounds} rounds",
        )

    # Store wandb run in context for access by visualization functions
    context.run_config["wandb_run"] = wandb_run

    # Add save_path and log_file_path to run config for clients (for client log paths)
    context.run_config["log_file_path"] = str(simulation_log_path)
    context.run_config["save_path"] = str(save_path)
    context.run_config["wandb_run_id"] = (
        run_id  # Pass shared run_id to clients for unified logging
    )

    # Save configuration snapshot
    import json

    # Get project version using standard importlib.metadata approach
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

    config_snapshot = {
        "timestamp": current_time.isoformat(),
        "run_config": dict(context.run_config),
        "federation": context.run_config.get("federation", "default"),
        "project_version": project_version,
        "output_structure": {
            "base_dir": str(save_path),
            "simulation_log": str(simulation_log_path),
            "config_file": str(save_path / "config.json"),
            "clients_dir": str(clients_dir),
            "server_dir": str(server_dir),
            "models_dir": str(models_dir),
        },
    }
    with open(save_path / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2, default=str)

    # Set global model initialization
    # Load a minimal dataset to get metadata for SmolVLA initialization
    from src.utils import load_lerobot_dataset
    from src.configs import DatasetConfig

    dataset_config = DatasetConfig.load()
    server_config = dataset_config.server[
        0
    ]  # Use server dataset for consistent initialization
    dataset = load_lerobot_dataset(server_config.name)
    dataset_meta = dataset.meta
    ndarrays = get_params(get_model(dataset_meta=dataset_meta))

    # 🛡️ VALIDATE: Server outgoing parameters (initial model)
    from src.utils import validate_and_log_parameters

    initial_param_hash = validate_and_log_parameters(ndarrays, "server_initial_model")

    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define strategy with evaluation aggregation
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Add evaluation configuration callback to provide save_path to clients
    eval_frequency = context.run_config.get("eval-frequency", 1)
    eval_batches = context.run_config.get("eval_batches", 0)
    logger.info(
        f"Server: Using eval_frequency={eval_frequency}, eval_batches={eval_batches}"
    )

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
        server_dir=server_dir,
        models_dir=models_dir,
        log_file=simulation_log_path,
        save_path=save_path,
        num_rounds=num_rounds,  # Pass total rounds for chart generation
        wandb_run=wandb_run,  # Pass wandb run for logging
        context=context,  # Pass context for checkpoint configuration
    )

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
