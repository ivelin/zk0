"""Unit tests for error handling scenarios in SmolVLA - focused on Flower API robustness."""

import pytest
from unittest.mock import patch, MagicMock

from src.client_app import SmolVLAClient


@pytest.mark.unit
class TestFitEvaluateExceptionHandling:
    """Test exception handling in fit and evaluate methods."""

    @pytest.fixture
    def client_config(self):
        """Default client configuration for tests."""
        mock_trainloader = MagicMock()
        mock_trainloader.dataset.meta.repo_id = "test/repo"

        return {
            "partition_id": 0,
            "local_epochs": 1,
            "trainloader": mock_trainloader,
            "nn_device": "cpu",
            "dataset_repo_id": "test/repo"
        }

    @pytest.fixture
    def mock_client(self, client_config):
        """Create a mock client for testing."""
        with patch('src.task.get_model') as mock_get_model:
            mock_get_model.return_value = MagicMock()
            client = SmolVLAClient(**client_config)
            return client
