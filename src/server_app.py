"""Flower server app for SmolVLA federated learning."""

import flwr as fl
from flwr.server import ServerApp, ServerConfig
import torch


def get_device(device_str: str = "auto"):
    """Get torch device from string specification."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda":
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def server_fn(context):
    """Server function factory for SmolVLA federated learning."""
    from flwr.server import ServerAppComponents
    import json
    import time
    from pathlib import Path

    # Get configuration from context (allows --run-config overrides)
    num_rounds = context.run_config.get("num-server-rounds", 50)

    # Create server configuration
    config = ServerConfig(
        num_rounds=num_rounds,
    )

    # Create output directory for server logs
    output_dir = Path("outputs") / "server_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Custom strategy with logging
    class LoggingFedAvg(fl.server.strategy.FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.round_metrics = []
            self.start_time = time.time()

        def aggregate_fit(self, server_round, results, failures):
            """Aggregate fit results and log progress."""
            aggregated_result = super().aggregate_fit(server_round, results, failures)

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
            progress_file = output_dir / "federation_progress.json"
            with open(progress_file, 'w') as f:
                json.dump({
                    "federation_start": self.start_time,
                    "current_round": server_round,
                    "total_rounds": num_rounds,
                    "rounds_completed": self.round_metrics,
                    "last_updated": time.time()
                }, f, indent=2)

            return aggregated_result

        def aggregate_evaluate(self, server_round, results, failures):
            """Aggregate evaluate results and log progress."""
            aggregated_result = super().aggregate_evaluate(server_round, results, failures)

            # Update final summary
            if server_round == num_rounds:
                summary_file = output_dir / "federation_summary.json"
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

    strategy = LoggingFedAvg()

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