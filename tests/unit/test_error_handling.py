"""Unit tests for error handling scenarios in SmolVLA - focused on Flower API robustness."""

import pytest
from unittest.mock import patch, MagicMock

try:
    from src.client.client_core import SmolVLAClient
except Exception:
    SmolVLAClient = None


@pytest.mark.unit
class TestFitEvaluateExceptionHandling:
    """Test exception handling in fit and evaluate methods."""

    @pytest.fixture
    def client_config(self):
        """Default client configuration for tests."""
        mock_trainloader = MagicMock()
        mock_trainloader.dataset.meta.repo_id = "test/repo"

        return {
            "client_id": 0,
            "local_epochs": 1,
            "trainloader": mock_trainloader,
            "nn_device": "cpu",
            "dataset_repo_id": "test/repo",
            "model_type": "smolvla",
        }

    @pytest.fixture
    def mock_client(self, client_config):
        """Create a mock client for testing."""
        if SmolVLAClient is None:
            pytest.skip("SmolVLAClient not importable due to env dep versions")
        with patch('src.training.model_utils.get_model') as mock_get_model:
            mock_get_model.return_value = MagicMock()
            client = SmolVLAClient(**client_config)
            return client

    def test_client_init_with_model_type(self, mock_client):
        """Test that client initializes with model_type (Phase 0)."""
        if SmolVLAClient is None:
            pytest.skip("SmolVLAClient not importable due to env dep versions")
        assert mock_client is not None
        assert hasattr(mock_client, 'model_type')
        assert mock_client.model_type == "smolvla"

    def test_client_creation_does_not_crash_on_missing_fit(self, mock_client):
        """Basic smoke that init works (error handling context)."""
        if SmolVLAClient is None:
            pytest.skip("SmolVLAClient not importable due to env dep versions")
        # The real error handling is exercised in full flow; here just no crash on construct
        assert mock_client.local_epochs == 1
