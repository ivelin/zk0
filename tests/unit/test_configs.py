"""Unit tests for configuration loading in src/configs/."""

import pytest
from unittest.mock import patch, mock_open
from src.configs.datasets import DatasetConfig, ClientConfig, ServerConfig


class TestDatasetConfig:
    """Test dataset configuration loading."""

    def test_client_config_creation(self):
        """Test ClientConfig dataclass creation."""
        config = ClientConfig(
            name="test/client",
            description="Test client",
            client_id=0
        )

        assert config.name == "test/client"
        assert config.description == "Test client"
        assert config.client_id == 0

    def test_server_config_creation(self):
        """Test ServerConfig dataclass creation."""
        config = ServerConfig(
            name="test/server",
            description="Test server",
            first_n_episodes_for_eval=10
        )

        assert config.name == "test/server"
        assert config.description == "Test server"
        assert config.first_n_episodes_for_eval == 10

    def test_dataset_config_creation(self):
        """Test DatasetConfig dataclass creation."""
        clients = [
            ClientConfig(name="client1", description="Client 1", client_id=0),
            ClientConfig(name="client2", description="Client 2", client_id=1)
        ]
        server = [
            ServerConfig(name="server1", description="Server 1", first_n_episodes_for_eval=5)
        ]

        config = DatasetConfig(clients=clients, server=server)

        assert len(config.clients) == 2
        assert len(config.server) == 1
        assert config.clients[0].name == "client1"
        assert config.server[0].first_n_episodes_for_eval == 5

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.configs.datasets.tomllib.load')
    def test_load_dataset_config(self, mock_tomllib_load, mock_file):
        """Test loading dataset config from pyproject.toml."""
        mock_tomllib_load.return_value = {
            "tool": {
                "zk0": {
                    "datasets": {
                        "clients": [
                            {
                                "name": "lerobot/svla_so100_stacking",
                                "description": "Stacking task",
                                "client_id": 0
                            }
                        ],
                        "server": [
                            {
                                "name": "lerobot/svla_so101_pickplace",
                                "description": "Server evaluation",
                                "first_n_episodes_for_eval": 10
                            }
                        ]
                    }
                }
            }
        }

        config = DatasetConfig.load()

        assert len(config.clients) == 1
        assert len(config.server) == 1
        assert config.clients[0].name == "lerobot/svla_so100_stacking"
        assert config.clients[0].client_id == 0
        assert config.server[0].first_n_episodes_for_eval == 10

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.configs.datasets.tomllib.load')
    def test_load_dataset_config_empty(self, mock_tomllib_load, mock_file):
        """Test loading dataset config with empty configuration."""
        mock_tomllib_load.return_value = {"tool": {"zk0": {}}}

        config = DatasetConfig.load()

        assert len(config.clients) == 0
        assert len(config.server) == 0

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.configs.datasets.tomllib.load')
    def test_load_dataset_config_missing_tool(self, mock_tomllib_load, mock_file):
        """Test loading dataset config when tool section is missing."""
        mock_tomllib_load.return_value = {}

        config = DatasetConfig.load()

        assert len(config.clients) == 0
        assert len(config.server) == 0

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_dataset_config_missing_file(self, mock_file):
        """Test loading dataset config when pyproject.toml is missing."""
        with pytest.raises(FileNotFoundError):
            DatasetConfig.load()