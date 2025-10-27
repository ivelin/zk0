"""zk0: A Flower / Hugging Face LeRobot app."""

from datetime import datetime
from pathlib import Path

from src.task import get_model, get_params, compute_param_norms, set_params




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










from src.logger import setup_server_logging
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
import numpy as np


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
        logger.info(
            f"Server: Instantiated - CUDA available: {torch.cuda.is_available()}"
        )

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
            import sys
            import traceback
            print(f"[DEBUG __init__] Model template creation failed: {e}", file=sys.stderr)
            print(f"[DEBUG __init__] Full traceback: {traceback.format_exc()}", file=sys.stderr)
            sys.stderr.flush()
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
        if should_skip_evaluation(server_round, self.eval_frequency):
            return None

        # Log CUDA before evaluation
        logger.info(
            f"Server: Starting evaluation round {server_round} - CUDA available: {torch.cuda.is_available()}"
        )

        logger.info(
            f"🔍 Server: _server_evaluate called for round {server_round} (frequency check passed)"
        )

        try:
            from src.configs import DatasetConfig
            from flwr.common import ndarrays_to_parameters

            # Store parameters for use in aggregate_fit if needed
            self.current_parameters = ndarrays_to_parameters(parameters)

            # Prepare model for evaluation
            model = prepare_evaluation_model(
                parameters, self.device, self.template_model
            )

            # Load dataset config for evaluation
            dataset_config = DatasetConfig.load()
            if not dataset_config.server:
                raise ValueError("No server evaluation dataset configured")

            # Perform composite evaluation across all server datasets
            eval_batches = self.context.run_config.get("eval_batches", 0)
            (
                composite_eval_loss,
                total_examples,
                composite_metrics,
                per_dataset_results,
            ) = evaluate_model_on_datasets(
                global_parameters=parameters,
                datasets_config=dataset_config.server,
                device=self.device,
                eval_batches=eval_batches,
            )

            # Use composite loss as the primary loss
            loss = composite_eval_loss
            num_examples = total_examples
            metrics = composite_metrics

            # Process evaluation metrics and update tracking
            process_evaluation_metrics(
                self,
                server_round,
                loss,
                metrics,
                self.last_aggregated_metrics,
                self.last_client_metrics,
            )

            # Log to WandB
            log_evaluation_to_wandb(
                self,
                server_round,
                loss,
                metrics,
                self.last_aggregated_metrics,
                self.last_client_metrics,
                per_dataset_results,
            )

            # Save evaluation results to file
            save_evaluation_results(
                self,
                server_round,
                loss,
                num_examples,
                metrics,
                self.last_aggregated_metrics,
                self.last_client_metrics,
            )

            # Generate chart on last round
            generate_evaluation_charts(self, server_round)

            # Finish WandB run after all logging is complete (always called on final round)
            if self.num_rounds and server_round == self.num_rounds:
                from src.wandb_utils import finish_wandb

                finish_wandb()
                logger.info("WandB run finished after final round")

            logger.info(
                f"✅ Server: _server_evaluate completed for round {server_round}"
            )
            return loss, metrics

        except Exception as e:
            logger.error(
                f"❌ Server: Failed _server_evaluate for round {server_round}: {e}"
            )
            logger.error(f"🔍 Detailed error: type={type(e).__name__}, args={e.args}")
            logger.exception("Full traceback in _server_evaluate")
            # Additional diagnostics: Check CUDA state post-failure
            if torch.cuda.is_available():
                logger.error(f"CUDA memory after eval failure: {torch.cuda.memory_summary()}")
            return None, {}

    def compute_fedprox_parameters(
        self, server_round: int, app_config: Dict[str, Scalar]
    ) -> Tuple[float, float]:
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
        if (
            dynamic_training_decay
            and hasattr(self, "server_eval_losses")
            and len(self.server_eval_losses) >= 3
        ):
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

                # 🔐 ADD: Use rounded hash for drift-resistant validation
                # Use float32 precision for hash (matches transmission dtype, minimal overhead)
                from src.utils import compute_rounded_hash

                server_computed_hash = compute_rounded_hash(
                    client_params, precision="float32"
                )
                logger.debug(
                    f"Server: Client {client_proxy.cid} rounded hash: {server_computed_hash}"
                )

                # Compare hashes (use rounded hash for drift resistance)
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
            current_mu, current_lr = self.compute_fedprox_parameters(
                server_round, app_config
            )

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
        from .server.server_utils import aggregate_client_metrics
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
            from .server.server_utils import compute_server_param_update_norm
            param_update_norm = compute_server_param_update_norm(
                self.previous_parameters, aggregated_parameters
            )
            aggregated_client_metrics["param_update_norm"] = param_update_norm

        # Store for use in _server_evaluate
        self.last_aggregated_metrics = aggregated_client_metrics

        # Store individual client metrics for detailed reporting
        from .server.server_utils import collect_individual_client_metrics
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
                    logger.exception("Traceback in save_model_checkpoint")

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
                        logger.exception("Traceback in final _server_evaluate")

                    # Push to Hugging Face Hub if configured (always attempt, even if save failed)
                    hf_repo_id = self.context.run_config.get("hf_repo_id")
                    if hf_repo_id:
                        try:
                            logger.info(
                                f"🚀 Server: Pushing final model to Hugging Face Hub: {hf_repo_id}"
                            )
                            self.push_model_to_hub(
                                aggregated_parameters, server_round, hf_repo_id
                            )
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
                        logger.info(
                            "ℹ️ Server: No hf_repo_id configured, skipping Hub push"
                        )

                except Exception as e:
                    logger.error(f"❌ Server: Failed to save final model: {e}")
                    # Still attempt Hub push even if checkpoint save failed
                    hf_repo_id = self.context.run_config.get("hf_repo_id")
                    if hf_repo_id:
                        try:
                            logger.info(
                                f"🚀 Server: Attempting Hub push despite checkpoint save failure: {hf_repo_id}"
                            )
                            self.push_model_to_hub(
                                aggregated_parameters, server_round, hf_repo_id
                            )
                            logger.info(
                                "✅ Server: Model pushed to Hub successfully despite checkpoint failure"
                            )
                        except Exception as push_e:
                            logger.error(
                                f"❌ Server: Both checkpoint save and Hub push failed: {push_e}"
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

        # Final logging before return to catch any late exceptions
        logger.info(f"✅ Server: aggregate_fit completing for round {server_round} - returning {aggregated_parameters is not None}")
        if self.early_stopping_triggered:
            logger.info(f"Early stopping active - final params from round {server_round}")
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
            logger.info(f"💾 Starting checkpoint save for round {server_round}")

            # Convert Flower Parameters to numpy arrays
            from flwr.common import parameters_to_ndarrays

            ndarrays = parameters_to_ndarrays(parameters)
            logger.debug(f"Converted {len(ndarrays)} parameters to ndarrays")

            # Create a state dict from the numpy arrays
            # Use the reusable template model for parameter names
            model = self.template_model

            # Create state dict with proper parameter names
            state_dict = {}
            conversion_errors = []
            for i, ((name, original_param), ndarray) in enumerate(
                zip(model.state_dict().items(), ndarrays)
            ):
                try:
                    # Convert numpy array back to torch tensor
                    tensor = torch.from_numpy(ndarray)

                    # Always convert to the original dtype to handle dtype drift from FedProx/aggregation
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
                if (
                    len(conversion_errors) > len(state_dict) * 0.1
                ):  # More than 10% failed
                    raise RuntimeError(
                        f"Too many parameter conversion errors ({len(conversion_errors)}/{len(ndarrays)})"
                    )

            # Save using safetensors format
            checkpoint_path = (
                models_dir / f"checkpoint_round_{server_round}.safetensors"
            )
            save_file(state_dict, checkpoint_path)

            # Log success with details
            checkpoint_size = (
                checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
            )
            logger.info(
                f"✅ Model checkpoint saved: {checkpoint_path} ({checkpoint_size} bytes)"
            )
            logger.info(f"📊 Checkpoint contains {len(state_dict)} parameter tensors")

        except Exception as e:
            logger.error(
                f"❌ Failed to save model checkpoint for round {server_round}: {e}"
            )
            logger.error(f"🔍 Error type: {type(e).__name__}, details: {e.args}")
            logger.exception("Full traceback in save_model_checkpoint")
            raise

    def push_model_to_hub(self, parameters, server_round: int, hf_repo_id: str) -> None:
        """Push model checkpoint to Hugging Face Hub.

        Args:
            parameters: Flower Parameters object containing model weights
            server_round: Current server round number
            hf_repo_id: Hugging Face repository ID (e.g., "username/repo-name")
        """
        try:
            logger.info(
                f"🚀 Starting HF Hub push for round {server_round} to {hf_repo_id}"
            )

            # Convert Flower Parameters to numpy arrays
            from flwr.common import parameters_to_ndarrays

            ndarrays = parameters_to_ndarrays(parameters)
            logger.debug(f"Converted {len(ndarrays)} parameters for HF push")

            # Create a state dict from the numpy arrays
            # Use the reusable template model for parameter names
            model = self.template_model

            # Create state dict with proper parameter names
            state_dict = {}
            conversion_errors = []
            for i, ((name, original_param), ndarray) in enumerate(
                zip(model.state_dict().items(), ndarrays)
            ):
                try:
                    # Convert numpy array back to torch tensor
                    tensor = torch.from_numpy(ndarray)

                    # Always convert to the original dtype to handle dtype drift from FedProx/aggregation
                    if original_param.dtype != tensor.dtype:
                        logger.debug(
                            f"Converting param {name} for HF push from {tensor.dtype} to {original_param.dtype}"
                        )
                        tensor = tensor.to(original_param.dtype)

                    # Validate shape matches
                    if tensor.shape != original_param.shape:
                        raise ValueError(
                            f"Shape mismatch for {name}: {tensor.shape} vs {original_param.shape}"
                        )

                    state_dict[name] = tensor

                except Exception as param_e:
                    error_msg = (
                        f"Failed to convert param {i} ({name}) for HF push: {param_e}"
                    )
                    logger.error(f"❌ {error_msg}")
                    conversion_errors.append(error_msg)
                    # Continue with other parameters

            if conversion_errors:
                logger.warning(
                    f"⚠️ {len(conversion_errors)} parameter conversion errors during HF push for round {server_round}"
                )
                if (
                    len(conversion_errors) > len(state_dict) * 0.1
                ):  # More than 10% failed
                    raise RuntimeError(
                        f"Too many parameter conversion errors for HF push ({len(conversion_errors)}/{len(ndarrays)})"
                    )

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
            logger.exception("Full traceback in push_model_to_hub")
            raise










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
        f"✅ Server: Dataset '{dataset_name}' loaded successfully (episodes: {len(dataset) if hasattr(dataset, '__len__') else 'unknown'})"
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
    logger.info(f"✅ Server: Set shared FL parameters to policy instance")

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
    datasets_config: List[ServerConfig],
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
    from src.task import test, set_params
    from src.utils import load_lerobot_dataset
    from lerobot.policies.factory import make_policy
    import numpy as np

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
    composite_metrics["per_dataset_results"] = per_dataset_results

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
    logger.info(f"🔍 Server: Using cached template model for evaluation...")
    model = template_model
    logger.info(
        f"✅ Server: Template model ready (total params: {sum(p.numel() for p in model.parameters())}"
    )

    # Set parameters
    logger.info(f"🔍 Server: Setting parameters...")
    from src.task import set_params

    set_params(model, parameters)
    logger.info(f"✅ Server: Parameters set successfully")

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
    round_metrics = {
        "round": server_round,
        "num_clients": aggregated_client_metrics.get("num_clients", 0),
        "avg_policy_loss": metrics.get("policy_loss", 0.0),
        "avg_client_loss": aggregated_client_metrics.get("avg_client_loss", 0.0),
        "param_update_norm": aggregated_client_metrics.get("param_update_norm", 0.0),
    }
    strategy.federated_metrics_history.append(round_metrics)

    # Update early stopping tracking
    update_early_stopping_tracking(strategy, server_round, loss)

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
        from src.wandb_utils import log_wandb_metrics
        from src.utils import prepare_server_wandb_metrics

        # Use utility function to prepare WandB metrics with same structure as JSON files
        # This ensures WandB metrics structure matches JSON file structure
        wandb_metrics = prepare_server_wandb_metrics(
            server_round=server_round,
            server_loss=loss,
            server_metrics=metrics,
            aggregated_client_metrics=aggregated_client_metrics,
            individual_client_metrics=individual_client_metrics,
            per_dataset_results=per_dataset_results,
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
        from datetime import datetime

        # Fix metrics bug: Update round number in individual_client_metrics before saving
        for metric in individual_client_metrics:
            metric["round"] = server_round

        server_file = strategy.server_dir / f"round_{server_round}_server_eval.json"
        data = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "server_side",
            "loss": loss,
            "num_examples": num_examples,
            "metrics": metrics,
            "aggregated_client_metrics": aggregated_client_metrics,  # Consolidated aggregated metrics
            "individual_client_metrics": individual_client_metrics,  # Individual client metrics with IDs
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


def generate_evaluation_charts(strategy, server_round: int) -> None:
    """Generate evaluation charts on final round.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number
    """
    if strategy.num_rounds and server_round == strategy.num_rounds:
        try:
            from src.visualization import SmolVLAVisualizer

            policy_loss_history = aggregate_eval_policy_loss_history(
                strategy.server_dir
            )
            visualizer = SmolVLAVisualizer()
            visualizer.plot_eval_policy_loss_chart(
                policy_loss_history, strategy.server_dir, wandb_run=strategy.wandb_run
            )
            if strategy.federated_metrics_history:
                visualizer.plot_federated_metrics(
                    strategy.federated_metrics_history,
                    strategy.server_dir,
                    wandb_run=strategy.wandb_run,
                )

            logger.info("Eval charts generated for final round")

        except Exception as e:
            logger.error(f"Failed to generate eval charts: {e}")


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

                # Extract server policy loss from metrics (prefer composite_eval_loss if available)
                metrics = server_data.get("metrics", {})
                policy_loss = metrics.get("composite_eval_loss") or metrics.get(
                    "policy_loss"
                )
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
    import sys
    print("[DEBUG server_fn] Starting server_fn execution", file=sys.stderr)
    sys.stderr.flush()

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    print(f"[DEBUG server_fn] Initialized ServerConfig with {num_rounds} rounds", file=sys.stderr)
    sys.stderr.flush()

    logger.info(f"🔧 Server: Initializing with {num_rounds} rounds")

    # Create output directory given timestamp (use env var if available, else current time)
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(f"outputs/{folder_name}")
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG server_fn] Created output dir: {save_path}", file=sys.stderr)
    sys.stderr.flush()

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
    print(f"[DEBUG server_fn] Setting up logging at {simulation_log_path}", file=sys.stderr)
    sys.stderr.flush()
    setup_server_logging(simulation_log_path)
    logger.info("Server logging initialized")
    print("[DEBUG server_fn] Logging setup complete", file=sys.stderr)
    sys.stderr.flush()

    # Load environment variables from .env file
    print("[DEBUG server_fn] Loading .env", file=sys.stderr)
    sys.stderr.flush()
    try:
        from dotenv import load_dotenv

        load_dotenv()
        print("[DEBUG server_fn] .env loaded successfully", file=sys.stderr)
        sys.stderr.flush()
        logger.debug("Environment variables loaded from .env file")
    except ImportError as e:
        print(f"[DEBUG server_fn] .env load failed (ImportError): {e}", file=sys.stderr)
        sys.stderr.flush()
        logger.debug("python-dotenv not available, skipping .env loading")
    except Exception as e:
        print(f"[DEBUG server_fn] .env load failed: {e}", file=sys.stderr)
        sys.stderr.flush()

    # Get wandb configuration from pyproject.toml
    from src.utils import get_tool_config

    flwr_config = get_tool_config("flwr", "pyproject.toml")
    app_config = flwr_config.get("app", {}).get("config", {})

    # Add app-specific configs to context.run_config for strategy access
    context.run_config["checkpoint_interval"] = app_config.get("checkpoint_interval", 2)

    # Initialize WandB if enabled
    print("[DEBUG server_fn] Checking WandB config", file=sys.stderr)
    sys.stderr.flush()
    from src.wandb_utils import init_server_wandb

    wandb_run = None
    run_id = f"zk0-sim-fl-run-{folder_name}"
    use_wandb = app_config.get("use-wandb", False)
    print(f"[DEBUG server_fn] use-wandb={use_wandb}", file=sys.stderr)
    sys.stderr.flush()
    if use_wandb:
        try:
            print("[DEBUG server_fn] Initializing WandB", file=sys.stderr)
            sys.stderr.flush()
            wandb_run = init_server_wandb(
                project="zk0",
                run_id=run_id,
                config=dict(app_config),
                dir=str(save_path),
                notes=f"Federated Learning Server - {num_rounds} rounds",
            )
            print("[DEBUG server_fn] WandB initialized successfully", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"[DEBUG server_fn] WandB init failed: {e}", file=sys.stderr)
            sys.stderr.flush()
            wandb_run = None
    else:
        print("[DEBUG server_fn] Skipping WandB (disabled)", file=sys.stderr)
        sys.stderr.flush()

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
    print("[DEBUG server_fn] Loading DatasetConfig", file=sys.stderr)
    sys.stderr.flush()
    # Load a minimal dataset to get metadata for SmolVLA initialization
    from src.utils import load_lerobot_dataset
    from src.configs import DatasetConfig

    try:
        dataset_config = DatasetConfig.load()
        print("[DEBUG server_fn] DatasetConfig loaded", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(f"[DEBUG server_fn] DatasetConfig.load failed: {e}", file=sys.stderr)
        sys.stderr.flush()
        raise

    if not dataset_config.server:
        print("[DEBUG server_fn] No server datasets configured - aborting", file=sys.stderr)
        sys.stderr.flush()
        raise ValueError("No server evaluation dataset configured")

    server_config = dataset_config.server[0]  # Use server dataset for consistent initialization
    print(f"[DEBUG server_fn] Loading server dataset: {server_config.name}", file=sys.stderr)
    sys.stderr.flush()

    try:
        dataset = load_lerobot_dataset(server_config.name)
        print("[DEBUG server_fn] Dataset loaded successfully", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(f"[DEBUG server_fn] load_lerobot_dataset failed for {server_config.name}: {e}", file=sys.stderr)
        sys.stderr.flush()
        raise

    dataset_meta = dataset.meta
    print("[DEBUG server_fn] Getting initial model params", file=sys.stderr)
    sys.stderr.flush()

    try:
        ndarrays = get_params(get_model(dataset_meta=dataset_meta))
        print(f"[DEBUG server_fn] Initial params obtained: {len(ndarrays)} arrays", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(f"[DEBUG server_fn] get_params/get_model failed: {e}", file=sys.stderr)
        sys.stderr.flush()
        raise

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

    print("[DEBUG server_fn] Creating AggregateEvaluationStrategy", file=sys.stderr)
    sys.stderr.flush()

    try:
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
        print("[DEBUG server_fn] Strategy created successfully", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(f"[DEBUG server_fn] AggregateEvaluationStrategy creation failed: {e}", file=sys.stderr)
        sys.stderr.flush()
        import traceback
        print(f"[DEBUG server_fn] Full traceback: {traceback.format_exc()}", file=sys.stderr)
        sys.stderr.flush()
        raise

    print("[DEBUG server_fn] Returning ServerAppComponents", file=sys.stderr)
    sys.stderr.flush()

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
