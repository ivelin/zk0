"""Dataset configuration for zk0."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class ClientConfig:
    """Configuration for a single FL client."""

    name: str
    description: str
    client_id: int


@dataclass
class ServerConfig:
    """Configuration for server evaluation dataset."""

    name: str
    description: str
    evaldata_id: int


@dataclass
class DatasetConfig:
    """Configuration for all FL datasets."""

    clients: List[ClientConfig]
    server: List[ServerConfig]

    @classmethod
    def load(cls) -> "DatasetConfig":
        """Load dataset configuration from pyproject.toml.

        Note: In production mode, clients ignore partitions and load datasets
        directly from run_config (dataset.repo_id or dataset.root) instead of
        using the clients list here. This method is used for simulation mode
        clients and server evaluation datasets in both modes.
        """
        # pyproject.toml is in the current working directory
        pyproject_path = Path("pyproject.toml")

        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)

        # Extract datasets configuration
        datasets_config = (
            pyproject_data.get("tool", {}).get("zk0", {}).get("datasets", {})
        )
        clients_data = datasets_config.get("clients", [])
        server_data = datasets_config.get("server", [])

        # Convert to config objects
        clients = []
        for client_data in clients_data:
            clients.append(ClientConfig(**client_data))

        server = []
        for server_item in server_data:
            server.append(ServerConfig(**server_item))

        return cls(clients=clients, server=server)
