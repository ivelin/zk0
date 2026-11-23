"""AggregateEvaluationStrategy for zk0 federated learning."""

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
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx
from loguru import logger
from safetensors.torch import save_file

from src.training.model_utils import compute_param_norms, get_model, set_params
from src.common.parameter_utils import compute_parameter_hash
from src.common.utils import get_tool_config, load_lerobot_dataset
from .wandb_utils import finish_wandb
from .metrics_utils import aggregate_and_log_metrics
from .metrics_utils import finalize_round_metrics
from .model_checkpointing import save_and_push_model
from .evaluation import (
    evaluate_model_on_datasets,
    should_skip_evaluation,
    prepare_evaluation_model,
    process_evaluation_metrics,
    log_evaluation_to_wandb,
    save_evaluation_results,
)


def dynamic_weighted_average(entries):
    """Registered dynamic weighting (Study 001 adaptation #6, D-11/G-3).

    Pure function; unit-checked against the study driver's
    aggregate_round (fl_driver.py) on identical synthetic inputs.
    entries: list of (ndarrays, num_examples, policy_loss).
    Rule (the driver's mirror of network.py aggregate_fit): median of
    the accepted losses; f_i = clip(median / L_i, 0.5, 2.0) when
    L_i > 0 and median > 0, else 1.0; eff_i = round(num_examples_i *
    f_i); weighted average with eff_i. Accumulation here is fp32
    (registered substrate difference (iii) in D-11); the driver
    accumulates in fp64 and casts to fp32.
    Returns (aggregated_ndarrays, record_dict).
    """
    losses = [float(loss) for _, _, loss in entries]
    med = float(np.median(losses)) if losses else 0.0
    per_client = []
    weighted = []
    total_eff = 0
    for idx, (arrays, num_examples, loss) in enumerate(entries):
        if loss > 0 and med > 0:
            f = float(np.clip(med / loss, 0.5, 2.0))
        else:
            f = 1.0
        eff = int(round(int(num_examples) * f))
        total_eff += eff
        weighted.append((arrays, eff))
        per_client.append(
            {
                "client_index": idx,
                "n_examples": int(num_examples),
                "policy_loss": float(loss),
                "weight_factor": round(f, 4),
                "effective_weight": eff,
            }
        )
    if not weighted or total_eff == 0:
        raise RuntimeError("dynamic_weighted_average: no accepted updates")
    aggregated = None
    for arrays, eff in weighted:
        if aggregated is None:
            aggregated = [a.astype(np.float32) * eff for a in arrays]
        else:
            for j in range(len(aggregated)):
                aggregated[j] += arrays[j].astype(np.float32) * eff
    aggregated = [a / total_eff for a in aggregated]
    record = {
        "median_policy_loss": med,
        "total_effective_weight": total_eff,
        "per_client": per_client,
    }
    return aggregated, record


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
        # Study 001 adaptation #10 (D-11/G-7): resume wiring - when a
        # run is relaunched from a round checkpoint, round_offset
        # shifts every round-indexed quantity (fit-config round,
        # lr_at offsets, checkpoint names) onto the registered
        # trajectory. Read from run_config (pyproject default 0, or a
        # --run-config override at resume time).
        self.round_offset = (
            int(context.run_config.get("round_offset", 0)) if context is not None else 0
        )
        if self.round_offset:
            logger.info(
                f"Server: round_offset={self.round_offset} (resume segment; rounds numbered from {self.round_offset + 1})"
            )
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

        logger.info(
            f"AggregateEvaluationStrategy: Initialized with proximal_mu={proximal_mu}, eval_frequency={self.eval_frequency}"
        )

        # Override evaluate_fn for server-side evaluation (called by strategy.evaluate every round, gated by frequency)
        # This replaces the default None evaluate_fn to enable server-side eval via Flower's standard flow
        self.evaluate_fn = self._server_evaluate

        # Create reusable model template for parameter name extraction (server is stateful, no race conditions)
        # The template's metadata comes from the TRAINING data (first
        # client dataset), never from a server evaluation dataset:
        # the template declares the input_features written into every
        # saved checkpoint (Study 001 Incident 1).
        try:
            from src.server.server_utils import load_template_dataset_meta

            dataset_meta = load_template_dataset_meta()
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
                # Server evaluation is optional and off unless a
                # server dataset is configured (Study 001 gap G-5:
                # the substrate must run cleanly with NO server
                # entry - a leftover entry here was Incident 1's
                # root cause, so absence is the safe default).
                logger.info(
                    "Server: no server evaluation dataset configured - "
                    "skipping server evaluation for round "
                    f"{server_round}"
                )
                return None

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

            # Store per-dataset results for checkpoint metadata,
            # stamped with the round they were measured in (gap
            # G-10: prepare_server_eval_metrics reports them only
            # for that round, never carried into later rounds).
            self.last_per_dataset_results = per_dataset_results
            self.last_per_dataset_results_round = server_round

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

            # Generate chart on last round using in-memory data
            if self.num_rounds and server_round == self.num_rounds:
                try:
                    # Use in-memory data instead of reading JSON files
                    policy_loss_history = {}
                    if hasattr(self, "server_eval_losses") and self.server_eval_losses:
                        # Label each loss with the round it was
                        # MEASURED in (gap G-10); enumerating the
                        # list mislabels rounds whenever evaluations
                        # are sparse or off.
                        eval_rounds = getattr(self, "server_eval_rounds", None)
                        for idx, loss in enumerate(self.server_eval_losses):
                            round_idx = (
                                eval_rounds[idx]
                                if eval_rounds and len(eval_rounds) == len(self.server_eval_losses)
                                else idx
                            )
                            policy_loss_history[round_idx] = {
                                "server_policy_loss": float(loss),
                                "action_dim": 7,  # Default
                            }

                    from .visualization import SmolVLAVisualizer
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

                    logger.info("Eval charts generated for final round")

                except Exception as e:
                    logger.error(f"Failed to generate eval charts: {e}")

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

        # Study 001 adaptation #5 (D-11/G-2): fixed FedProx path - the
        # registered mu is a CONSTANT 0.1 for all 30 rounds. Neither
        # stock path ever returns it (the fallback halves mu every 10
        # rounds; the dynamic path scales it by an eval-trend factor),
        # so when the app config carries fixed_fedprox = true, return
        # (strategy.proximal_mu, app_config initial_lr) unchanged.
        if app_config.get("fixed_fedprox", False):
            fixed_lr = app_config.get("initial_lr", 1e-3)
            logger.info(
                f"Server R{server_round}: fixed_fedprox mu={initial_mu:.6f}, lr={fixed_lr:.6f} (constant, registered)"
            )
            return initial_mu, fixed_lr

        # Check if dynamic training decay is enabled
        dynamic_training_decay = app_config.get("dynamic_training_decay", False)
        if (
            dynamic_training_decay
            and hasattr(self, "server_eval_losses")
            and len(self.server_eval_losses) >= 3
        ):
            from src.training.scheduler_utils import compute_joint_adjustment

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
                from src.common.parameter_utils import compute_rounded_hash

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
        # Study 001 adaptation #10 (D-11/G-7): on a resumed segment,
        # number rounds on the registered trajectory.
        server_round = server_round + self.round_offset
        # Log CUDA before training round
        logger.info(
            f"Server: Starting training round {server_round} - CUDA available: {torch.cuda.is_available()}"
        )

        logger.info(f"Server: Configuring fit for round {server_round}")

        # Get configuration from pyproject.toml
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
            # FedProx: Dynamically adjust proximal_mu and LR based on evaluation trends
            current_mu, current_lr = self.compute_fedprox_parameters(
                server_round, app_config
            )

            updated_fit_config["proximal_mu"] = current_mu

            # Study 001 adaptation #7 (D-11/G-4): registered LR
            # schedule - the round's constant LR is lr_at evaluated
            # at the MID-ROUND offset of the node's 3000-step budget
            # (100 local steps per round), replacing the stock static
            # overwrite that discarded the computed per-round LR.
            from src.server.lr_schedule import lr_at

            round_lr = lr_at((server_round - 1) * 100 + 50)
            updated_fit_config["initial_lr"] = round_lr
            logger.debug(
                f"Server: Added initial_lr={round_lr} (lr_at mid-round offset {(server_round - 1) * 100 + 50}) to client {client_proxy.cid} config"
            )

            # + Add use_wandb flag for client-side WandB enablement
            use_wandb = app_config.get("use-wandb", False)
            updated_fit_config["use_wandb"] = use_wandb
            logger.debug(
                f"Server: Added use_wandb={use_wandb} to client {client_proxy.cid} config"
            )

            # Add batch_size from run_config (command line) as primary, no fallback to app_config
            batch_size = self.context.run_config.get("batch_size", 64)
            updated_fit_config["batch_size"] = batch_size
            logger.debug(
                f"Server: Added batch_size={batch_size} to client {client_proxy.cid} config"
            )

            # WandB run_id not needed in fit config - client already initialized wandb in client_fn

            # 🛡️ VALIDATE: Server outgoing parameters (for training) - with detailed logging
            from src.common.utils import validate_and_log_parameters

            fit_ndarrays = parameters_to_ndarrays(fit_ins.parameters)
            logger.debug(
                f"Server: Pre-serialization params for client {client_proxy.cid}: {len(fit_ndarrays)} arrays"
            )

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

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate client evaluation results (policy_loss only)."""

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

    def aggregate_parameters(self, server_round: int, validated_results):
        """Aggregate parameters from validated client results.

        Study 001 adaptation #6 (D-11/G-3): the registered dynamic
        weighting replaces the FedProx parent's plain num_examples
        FedAvg - eff_i = round(n_examples_i * clip(median(L)/L_i,
        0.5, 2.0)), where L_i is the client's returned policy_loss
        (NOT the substrate's total loss, which includes the FedProx
        term) and the median is over the round's valid fits.
        Per-round per-client (n_examples, policy_loss, factor,
        eff_weight) is logged and written to
        <save_path>/server/aggregation_round_<round>.json - the
        section 9.2 re-derivation is impossible without this record.

        Args:
            server_round: Current server round number (registered
                numbering; aggregate_fit applies the round offset)
            validated_results: List of validated (client_proxy, fit_result) tuples

        Returns:
            tuple: (aggregated_parameters, parent_metrics)
        """
        logger.info(
            f"Server: Aggregating parameters for round {server_round} with registered dynamic weighting (adaptation #6), {len(validated_results)} validated clients"
        )
        entries = []
        client_ids = []
        for client_proxy, fit_res in validated_results:
            metrics = fit_res.metrics or {}
            loss = metrics.get("policy_loss")
            loss = float(loss) if loss is not None else 0.0
            entries.append(
                (
                    parameters_to_ndarrays(fit_res.parameters),
                    int(fit_res.num_examples),
                    loss,
                )
            )
            client_ids.append(metrics.get("client_id"))
        aggregated_ndarrays, record = dynamic_weighted_average(entries)
        for row, cid in zip(record["per_client"], client_ids):
            row["client_id"] = cid
            logger.info(
                f"Server R{server_round} AGG: client_id={cid} n_examples={row['n_examples']} policy_loss={row['policy_loss']:.6f} factor={row['weight_factor']} eff_weight={row['effective_weight']}"
            )
        logger.info(
            f"Server R{server_round} AGG: median_policy_loss={record['median_policy_loss']:.6f} total_effective_weight={record['total_effective_weight']}"
        )
        try:
            import json as _json
            from pathlib import Path as _Path

            record_path = (
                _Path(self.save_path) / "server" / f"aggregation_round_{server_round}.json"
            )
            record_path.parent.mkdir(parents=True, exist_ok=True)
            with open(record_path, "w") as f:
                _json.dump({"round": server_round, **record}, f, indent=2)
            logger.info(f"Server R{server_round} AGG: record written to {record_path}")
        except Exception as rec_e:
            logger.warning(
                f"Server R{server_round} AGG: could not write aggregation record: {rec_e}"
            )
        return ndarrays_to_parameters(aggregated_ndarrays), {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results from clients."""
        # Study 001 adaptation #10 (D-11/G-7): on a resumed segment,
        # number rounds on the registered trajectory (aggregation
        # records and checkpoint names use the registered round).
        server_round = server_round + self.round_offset
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
        if len(validated_results) == 0:
            logger.warning(f"⚠️ Server: No valid fits in round {server_round} - retaining current params")
            if self.current_parameters is None:
                raise ValueError(f"No params available for empty round {server_round}")
            return self.current_parameters, {}

        # Aggregate parameters using FedProx strategy
        aggregated_parameters, parent_metrics = self.aggregate_parameters(
            server_round, validated_results
        )

        # Aggregate client metrics and compute norms

        aggregated_client_metrics = aggregate_and_log_metrics(
            self, server_round, validated_results, aggregated_parameters
        )

        # Finalize round metrics

        metrics = finalize_round_metrics(
            self, server_round, aggregated_client_metrics, parent_metrics
        )

        # Store the aggregated parameters for server-side evaluation
        self.current_parameters = aggregated_parameters

        # Store previous parameters for next round's update norm calculation
        self.previous_parameters = aggregated_parameters

        # CRITICAL: Always return valid parameters tuple to prevent Flower unpacking errors
        # This fixes the "cannot unpack non-iterable NoneType object" error
        if aggregated_parameters is None:
            logger.warning(
                f"⚠️ Server: No parameters aggregated for round {server_round}, returning initial parameters"
            )
            aggregated_parameters = self.initial_parameters

        # Log post-aggregation global norms (aggregated_parameters is guaranteed non-None after fallback)
        # Import here to avoid scope/shadowing issues

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

        # Log parameter update information (aggregated_parameters is guaranteed non-None)
        logger.info(
            f"✅ Server: Successfully aggregated parameters from {len(results)} clients for round {server_round}"
        )

        # 🛡️ VALIDATE: Server aggregated parameters
        from src.common.utils import validate_and_log_parameters

        validate_and_log_parameters(
            aggregated_ndarrays, f"server_aggregated_r{server_round}"
        )

        # Save model checkpoint and push to Hub if applicable
        # save_and_push_model already imported from .model_checkpointing

        save_and_push_model(self, server_round, aggregated_parameters, metrics)

        # Final logging before return to catch any late exceptions
        logger.info(
            f"✅ Server: aggregate_fit completing for round {server_round}/{self.num_rounds if self.num_rounds else 'unknown'} - returning {aggregated_parameters is not None}"
        )
        logger.info(f"🔄 Server: Round {server_round} complete. Total configured rounds: {self.num_rounds}. Proceeding to next if any.")
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

    def save_flower_history(self, history, output_dir):
        """Save Flower metrics history to JSON and generate plot.

        This method should be called after server.fit() completes, typically from
        the training script (e.g., train-fl-simulation.sh) with the returned history.

        Args:
            history: Flower History object from server.fit()
            output_dir: Directory to save files (e.g., outputs/YYYY-MM-DD_HH-MM-SS)
        """
        import json
        from pathlib import Path

        output_dir = Path(output_dir)
        flower_metrics = {
            "metrics_distributed_fit": history.metrics_distributed_fit,
            "metrics_centralized": history.metrics_centralized,
            "losses_distributed": history.losses_distributed,
            "losses_centralized": history.losses_centralized,
        }

        # Save to JSON
        with open(output_dir / "flower_history.json", "w") as f:
            json.dump(flower_metrics, f, indent=2)
        logger.info(f"✅ Saved Flower history to {output_dir / 'flower_history.json'}")

        # Generate plot
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            history.plot(ax=ax)
            ax.set_title("Federated Learning Convergence")
            ax.set_xlabel("Round")
            ax.set_ylabel("Loss")
            fig.savefig(output_dir / "flower_history_plot.png", dpi=150, bbox_inches='tight')
            logger.info(f"✅ Saved Flower history plot to {output_dir / 'flower_history_plot.png'}")
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot generation")