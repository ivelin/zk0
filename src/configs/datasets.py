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
    last_n_episodes_for_eval: int


@dataclass
class DatasetConfig:
    """Configuration for all FL datasets."""
    clients: List[ClientConfig]

    @classmethod
    def load(cls) -> "DatasetConfig":
        """Load dataset configuration from pyproject.toml."""
        # pyproject.toml is in the current working directory
        pyproject_path = Path("pyproject.toml")

        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)

        # Extract datasets configuration
        datasets_config = pyproject_data.get("tool", {}).get("zk0", {}).get("datasets", {})
        clients_data = datasets_config.get("clients", [])

        # Convert to ClientConfig objects
        clients = []
        for client_data in clients_data:
            clients.append(ClientConfig(**client_data))

        return cls(clients=clients)