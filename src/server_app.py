"""Flower server app for SmolVLA federated learning."""

import flwr as fl
from flwr.server import ServerApp, ServerConfig
import torch
import numpy as np
from pathlib import Path
from datetime import datetime


def get_device(device_str: str = "auto"):
    """Get torch device from string specification."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda":
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_evaluate_fn_callback(save_path: Path):
    """Return evaluation function that saves global model checkpoints."""
    def evaluate_fn(server_round: int, parameters, config):
        # Save global model checkpoint every round
        try:
            from pathlib import Path
            import torch
            import json
            from datetime import datetime

            # Create checkpoint directory for this round
            checkpoint_dir = save_path / "global_model" / f"round_{server_round}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save parameters as numpy arrays
            if parameters and hasattr(parameters, 'tensors'):
                params_file = checkpoint_dir / "parameters.npy"
                np.save(params_file, parameters.tensors)

                # Save metadata
                metadata = {
                    "round": server_round,
                    "timestamp": datetime.now().isoformat(),
                    "num_parameters": len(parameters.tensors) if parameters.tensors else 0,
                    "parameter_shapes": [p.shape for p in parameters.tensors] if parameters.tensors else []
                }

                with open(checkpoint_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                print(f"Global model checkpoint saved: {checkpoint_dir}")

        except Exception as e:
            print(f"Failed to save global checkpoint: {e}")

        return 0.0, {}  # Return dummy loss and metrics

    return evaluate_fn


def get_evaluate_config_callback(save_path: Path):
    """Return function to configure evaluation rounds."""
    def evaluate_config_fn(server_round: int):
        eval_save_path = save_path / "evaluate" / f"round_{server_round}"
        return {
            "save_path": str(eval_save_path),
            "round": server_round
        }
    return evaluate_config_fn


def server_fn(context):
    """Server function factory for SmolVLA federated learning."""
    from flwr.server import ServerAppComponents
    import json
    import time
    from pathlib import Path
    from datetime import datetime

    # Get configuration from context (allows --run-config overrides)
    num_rounds = context.run_config.get("num-server-rounds", 50)

    # Create organized server output directory
    server_output_dir = Path("outputs/server")
    server_output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped subdirectory for this run
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = server_output_dir / folder_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Create server configuration
    config = ServerConfig(num_rounds=num_rounds)

    # Custom strategy with logging
    class LoggingFedAvg(fl.server.strategy.FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.round_metrics = []
            self.start_time = time.time()
            self.save_path = save_path

        def aggregate_fit(self, server_round, results, failures):
            """Aggregate fit results and log progress."""
            aggregated_result = super().aggregate_fit(server_round, results, failures)

            # aggregated_result is a tuple (parameters, metrics) in Flower
            if isinstance(aggregated_result, tuple) and len(aggregated_result) >= 1:
                parameters = aggregated_result[0]
                if parameters is not None:
                    self._save_global_checkpoint(server_round, parameters)

            # Log round metrics
            round_data = {
                "round": server_round,
                "num_clients": len(results),
                "num_failures": len(failures),
                "timestamp": time.time(),
                "elapsed_time": time.time() - self.start_time,
            }

            # Extract metrics from client results
            client_metrics = []
            for client_result in results:
                if hasattr(client_result, 'metrics') and client_result.metrics:
                    client_metrics.append(client_result.metrics)

            if client_metrics:
                round_data["client_metrics"] = client_metrics

            self.round_metrics.append(round_data)

            # Save progress to file
            progress_file = self.save_path / "federation_progress.json"
            with open(progress_file, 'w') as f:
                json.dump({
                    "federation_start": self.start_time,
                    "current_round": server_round,
                    "total_rounds": num_rounds,
                    "rounds_completed": self.round_metrics,
                    "last_updated": time.time()
                }, f, indent=2)

            return aggregated_result

        def _save_global_checkpoint(self, server_round, parameters):
            """Save the global model checkpoint after aggregation."""
            try:
                # Create checkpoint directory for this round
                checkpoint_dir = self.save_path / "global_model" / f"round_{server_round}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_path = checkpoint_dir / f"global_model_round_{server_round}.pt"
                checkpoint_data = {
                    'round': server_round,
                    'parameters': parameters.tensors if hasattr(parameters, 'tensors') else parameters,
                    'timestamp': time.time(),
                    'num_clients': len(self.round_metrics[-1]['client_metrics']) if self.round_metrics else 0,
                }

                # Save using torch
                torch.save(checkpoint_data, checkpoint_path)
                print(f"Global model checkpoint saved: {checkpoint_path}")

                # Also save as numpy arrays for easier inspection
                if hasattr(parameters, 'tensors'):
                    np_checkpoint_path = checkpoint_dir / f"global_model_round_{server_round}_params.npy"
                    np.save(np_checkpoint_path, parameters.tensors)
                    print(f"Global model parameters saved: {np_checkpoint_path}")

                # Save metadata
                metadata = {
                    "round": server_round,
                    "timestamp": datetime.now().isoformat(),
                    "num_parameters": len(parameters.tensors) if hasattr(parameters, 'tensors') and parameters.tensors else 0,
                    "parameter_shapes": [p.shape for p in parameters.tensors] if hasattr(parameters, 'tensors') and parameters.tensors else [],
                    "checkpoint_files": [str(checkpoint_path), str(np_checkpoint_path)]
                }

                with open(checkpoint_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

            except Exception as e:
                print(f"Failed to save global checkpoint: {e}")

        def aggregate_evaluate(self, server_round, results, failures):
            """Aggregate evaluate results and log progress."""
            aggregated_result = super().aggregate_evaluate(server_round, results, failures)

            # Update final summary
            if server_round == num_rounds:
                summary_file = self.save_path / "federation_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump({
                        "federation_completed": True,
                        "total_rounds": num_rounds,
                        "total_time": time.time() - self.start_time,
                        "final_metrics": aggregated_result,
                        "all_rounds": self.round_metrics,
                        "completion_timestamp": time.time()
                    }, f, indent=2)

            return aggregated_result

    strategy = LoggingFedAvg(
        on_evaluate_config_fn=get_evaluate_config_callback(save_path),
        evaluate_fn=get_evaluate_fn_callback(save_path),
    )

    return ServerAppComponents(
        strategy=strategy,
        config=config
    )


# Create server app with server function
app = ServerApp(server_fn=server_fn)


def main() -> None:
    """Run the SmolVLA federated learning server."""
    # Start server
    app.run()


if __name__ == "__main__":
    main()