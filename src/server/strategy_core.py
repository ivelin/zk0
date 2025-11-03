"""Core AggregateEvaluationStrategy class for zk0 federated learning."""

from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import (
    Context,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerAppComponents, ServerApp, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx
from loguru import logger
from safetensors.torch import save_file

from src.logger import setup_server_logging
from src.training.model_utils import compute_param_norms, get_model, get_params, set_params
from src.core.utils import get_tool_config, load_lerobot_dataset
from src.visualization import SmolVLAVisualizer
from src.wandb_utils import finish_wandb, init_server_wandb, log_wandb_metrics
from src.core.utils import prepare_server_wandb_metrics
from .parameter_validation import compute_fedprox_parameters
from .fit_configuration import configure_fit
from .aggregation import aggregate_parameters, aggregate_and_log_metrics
from .model_checkpointing import save_and_push_model, finalize_round_metrics
from .server_utils import get_runtime_mode
from .evaluation import (
    evaluate_model_on_datasets,
    should_skip_evaluation,
    prepare_evaluation_model,
    process_evaluation_metrics,
    log_evaluation_to_wandb,
    save_evaluation_results,
    generate_evaluation_charts,
)


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

        # Determine runtime mode
        self.mode = get_runtime_mode(context) if context else "unknown"

        logger.info(
            f"AggregateEvaluationStrategy: Initialized with proximal_mu={proximal_mu}, eval_frequency={self.eval_frequency}, mode={self.mode}"
        )

        # Override evaluate_fn for server-side evaluation (called by strategy.evaluate every round, gated by frequency)
        # This replaces the default None evaluate_fn to enable server-side eval via Flower's standard flow
        self.evaluate_fn = self._server_evaluate

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure the next round of training."""
        return configure_fit(self, server_round, parameters, client_manager)

        # Create reusable model template for parameter name extraction (server is stateful, no race conditions)
        try:
            from src.configs import DatasetConfig

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

            print(
                f"[DEBUG __init__] Model template creation failed: {e}", file=sys.stderr
            )
            print(
                f"[DEBUG __init__] Full traceback: {traceback.format_exc()}",
                file=sys.stderr,
            )
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

            # Store parameters for use in aggregate_fit if needed
            self.current_parameters = ndarrays_to_parameters(parameters)

            # Prepare model for evaluation
            prepare_evaluation_model(
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

            # Store per-dataset results for checkpoint metadata
            self.last_per_dataset_results = per_dataset_results

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
                logger.error(
                    f"CUDA memory after eval failure: {torch.cuda.memory_summary()}"
                )
            return None, {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results from clients."""
        from .parameter_validation import validate_client_parameters

        logger.info(f"Server: Aggregating fit results for round {server_round} (mode: {self.mode})")
        logger.info(
            f"📊 Server: Received {len(results)} successful results, {len(failures)} failures"
        )

        # Validate client parameters
        validated_results = validate_client_parameters(self, results)

        # Aggregate parameters
        aggregated_parameters = aggregate_parameters(self, validated_results, failures, server_round)

        # Aggregate and log client metrics
        aggregated_client_metrics = aggregate_and_log_metrics(self, validated_results, failures, server_round)

        # Store for server evaluation
        self.last_client_metrics = aggregated_client_metrics.get("individual_metrics", [])

        # Finalize round metrics
        aggregated_parameters, metrics = finalize_round_metrics(self, server_round, aggregated_parameters, aggregated_client_metrics)

        # Store parameters for next round
        self.current_parameters = aggregated_parameters
        self.previous_parameters = aggregated_parameters

        # Save model checkpoint and push to Hub if applicable
        save_and_push_model(self, server_round, aggregated_parameters, metrics)

        logger.info(
            f"✅ Server: aggregate_fit completed for round {server_round}"
        )
        return aggregated_parameters, metrics