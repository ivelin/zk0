"""Dataset configuration for zk0."""

import sys
from dataclasses import dataclass
from typing import List
from loguru import logger

if sys.version_info >= (3, 11):
    pass
else:
    pass


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
        """Load dataset configuration from pyproject.toml using utility.

        Note: In production mode, clients ignore partitions and load datasets
        directly from run_config (dataset.repo_id or dataset.root) instead of
        using the clients list here. This method is used for simulation mode
        clients and server evaluation datasets in both modes.
        """
        from src.common.utils import get_tool_config

        datasets_config = get_tool_config("zk0.datasets")
        logger.info(f"Raw datasets_config keys: {list(datasets_config.keys())}")
        logger.info(f"Raw datasets_config['server']: {datasets_config.get('server', 'MISSING')}")
        clients_data = datasets_config.get("clients", [])
        server_data = datasets_config.get("server", [])
        logger.info(f"Parsed server configs: {len(server_data)}")

        # Convert to config objects
        clients = []
        for client_data in clients_data:
            clients.append(ClientConfig(**client_data))

        server = []
        for server_item in server_data:
            server.append(ServerConfig(**server_item))

        return cls(clients=clients, server=server)
