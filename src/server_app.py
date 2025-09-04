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


# Create server app with configuration
app = ServerApp(
    config=ServerConfig(
        num_rounds=50,  # Reduced for initial testing
    )
)


def main() -> None:
    """Run the SmolVLA federated learning server."""
    # Start server
    app.run()


if __name__ == "__main__":
    main()