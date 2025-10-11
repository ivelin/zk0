"""Unit tests for WandB utilities in src/wandb_utils.py."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import os
import sys


class TestWandbUtils:
    """Test WandB utility functions."""

    @pytest.fixture(autouse=True)
    def mock_wandb_module(self):
        """Mock the wandb module for all tests."""
        mock_wandb = Mock()
        sys.modules['wandb'] = mock_wandb
        yield mock_wandb
        # Clean up
        if 'wandb' in sys.modules:
            del sys.modules['wandb']

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_key"})
    def test_init_wandb_with_api_key(self, mock_wandb_module):
        """Test WandB initialization when API key is available."""
        from src.wandb_utils import init_wandb

        mock_run = MagicMock()
        mock_run.name = "test_run"
        mock_run.id = "test_id"
        mock_run.project = "test_project"
        mock_wandb_module.init.return_value = mock_run

        result = init_wandb(project="test_project", name="test_name")

        mock_wandb_module.init.assert_called_once()
        assert result == mock_run

    @patch.dict(os.environ, {}, clear=True)
    def test_init_wandb_without_api_key(self, mock_wandb_module):
        """Test WandB initialization when API key is missing."""
        from src.wandb_utils import init_wandb

        result = init_wandb()

        mock_wandb_module.init.assert_not_called()
        assert result is None

    def test_init_wandb_import_error(self, mock_wandb_module):
        """Test WandB initialization when wandb is not installed."""
        from src.wandb_utils import init_wandb

        # Simulate import error by removing the module
        del sys.modules['wandb']

        result = init_wandb()

        assert result is None

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_key"})
    def test_log_wandb_metrics_with_active_run(self, mock_wandb_module):
        """Test logging metrics when WandB run is active."""
        from src.wandb_utils import log_wandb_metrics

        mock_wandb_module.run = MagicMock()
        mock_wandb_module.log = MagicMock()

        metrics = {"loss": 0.5, "accuracy": 0.9}
        log_wandb_metrics(metrics, step=10)

        mock_wandb_module.log.assert_called_once_with(metrics, step=10)

    def test_log_wandb_metrics_without_active_run(self, mock_wandb_module):
        """Test logging metrics when no WandB run is active."""
        from src.wandb_utils import log_wandb_metrics

        mock_wandb_module.run = None
        mock_wandb_module.log = MagicMock()

        metrics = {"loss": 0.5}
        log_wandb_metrics(metrics)

        mock_wandb_module.log.assert_not_called()

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_key"})
    def test_init_client_wandb_with_run_id(self, mock_wandb_module):
        """Test client WandB initialization with server run ID."""
        from src.wandb_utils import init_client_wandb

        mock_run = MagicMock()
        mock_run.name = "server_run"
        mock_run.id = "server_id"
        mock_wandb_module.init.return_value = mock_run

        result = init_client_wandb(
            partition_id=0,
            dataset_name="test/dataset",
            local_epochs=5,
            batch_size=32,
            wandb_run_id="server_run_id"
        )

        mock_wandb_module.init.assert_called_once()
        assert result == mock_run

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_key"})
    def test_init_client_wandb_without_run_id(self, mock_wandb_module):
        """Test client WandB initialization without server run ID."""
        from src.wandb_utils import init_client_wandb

        mock_run = MagicMock()
        mock_run.name = "client_run"
        mock_run.id = "client_id"
        mock_wandb_module.init.return_value = mock_run

        result = init_client_wandb(
            partition_id=1,
            dataset_name="test/dataset",
            local_epochs=10,
            batch_size=64
        )

        mock_wandb_module.init.assert_called_once()
        assert result == mock_run

    @patch.dict(os.environ, {}, clear=True)
    def test_init_client_wandb_without_api_key(self, mock_wandb_module):
        """Test client WandB initialization when API key is missing."""
        from src.wandb_utils import init_client_wandb

        result = init_client_wandb(
            partition_id=0,
            dataset_name="test/dataset",
            local_epochs=5,
            batch_size=32
        )

        mock_wandb_module.init.assert_not_called()
        assert result is None

    def test_finish_wandb_with_active_run(self, mock_wandb_module):
        """Test finishing WandB run when active."""
        from src.wandb_utils import finish_wandb

        mock_wandb_module.run = MagicMock()
        mock_wandb_module.finish = MagicMock()

        finish_wandb()

        mock_wandb_module.finish.assert_called_once()

    def test_finish_wandb_without_active_run(self, mock_wandb_module):
        """Test finishing WandB run when no active run."""
        from src.wandb_utils import finish_wandb

        mock_wandb_module.run = None
        mock_wandb_module.finish = MagicMock()

        finish_wandb()

        mock_wandb_module.finish.assert_not_called()

    def test_finish_wandb_import_error(self, mock_wandb_module):
        """Test finishing WandB run when wandb is not installed."""
        from src.wandb_utils import finish_wandb

        # Simulate import error by removing the module
        del sys.modules['wandb']

        # Should not raise exception
        finish_wandb()