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
        sys.modules["wandb"] = mock_wandb
        yield mock_wandb
        # Clean up
        if "wandb" in sys.modules:
            del sys.modules["wandb"]

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_key"})
    def test_init_server_wandb_with_api_key(self, mock_wandb_module):
        """Test WandB initialization when API key is available."""
        from src.wandb_utils import init_server_wandb

        mock_run = MagicMock()
        mock_run.name = "test_run"
        mock_run.id = "test_id"
        mock_run.project = "test_project"
        mock_wandb_module.init.return_value = mock_run

        result = init_server_wandb(project="test_project", name="test_name")

        mock_wandb_module.init.assert_called_once()
        assert result == mock_run

    @patch.dict(os.environ, {}, clear=True)
    @patch("dotenv.load_dotenv")
    def test_init_server_wandb_without_api_key(
        self, mock_load_dotenv, mock_wandb_module
    ):
        """Test WandB initialization when API key is missing."""
        from src.wandb_utils import init_server_wandb

        # Mock load_dotenv to do nothing (prevent .env file loading)
        mock_load_dotenv.return_value = None

        result = init_server_wandb()

        mock_wandb_module.init.assert_not_called()
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
        del sys.modules["wandb"]

        # Should not raise exception
        finish_wandb()

    def test_log_wandb_metrics_with_prefixed_client_metrics(self, mock_wandb_module):
        """Test logging prefixed client metrics to WandB (new pattern for server-side per-client logging)."""
        from src.wandb_utils import log_wandb_metrics

        mock_wandb_module.run = MagicMock()
        mock_wandb_module.log = MagicMock()

        # Sample prefixed client metrics as prepared in server_app.py
        prefixed_metrics = {
            "client_0_policy_loss": 0.5,
            "client_0_fedprox_loss": 0.1,
            "client_0_param_update_norm": 0.01,
            # ... other fields from individual_client_metrics
        }

        log_wandb_metrics(prefixed_metrics, step=5)

        mock_wandb_module.log.assert_called_once_with(prefixed_metrics, step=5)

    def test_log_wandb_metrics_with_aggregated_server_metrics(self, mock_wandb_module):
        """Test logging aggregated server metrics to WandB (extended for all aggregated_client_metrics fields)."""
        from src.wandb_utils import log_wandb_metrics

        mock_wandb_module.run = MagicMock()
        mock_wandb_module.log = MagicMock()

        # Sample aggregated metrics as prepared in server_app.py
        aggregated_metrics = {
            "server_avg_client_loss": 0.45,
            "server_std_client_loss": 0.05,
            "server_avg_client_proximal_loss": 0.02,
            "server_avg_client_grad_norm": 1.2,
            "server_num_clients": 4,
            "server_param_update_norm": 0.015,
        }

        log_wandb_metrics(aggregated_metrics, step=10)

        mock_wandb_module.log.assert_called_once_with(aggregated_metrics, step=10)

    def test_log_wandb_metrics_with_final_summary_metrics(self, mock_wandb_module):
        """Test logging final summary metrics including new cumulative fields (e.g., max across clients)."""
        from src.wandb_utils import log_wandb_metrics

        mock_wandb_module.run = MagicMock()
        mock_wandb_module.log = MagicMock()

        # Sample final metrics as prepared in server_app.py final round
        final_metrics = {
            "server_final_round": 30,
            "server_final_eval_loss": 0.3,
            "server_max_client_cumulative_epochs": 150.0,
            "server_total_client_cumulative_rounds": 120,
            # ... other final fields
        }

        log_wandb_metrics(final_metrics)

        mock_wandb_module.log.assert_called_once_with(final_metrics)
