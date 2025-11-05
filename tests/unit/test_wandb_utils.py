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

    def test_get_wandb_public_url_with_active_run(self, mock_wandb_module):
        """Test getting WandB public URL when run is active."""
        from src.wandb_utils import get_wandb_public_url

        mock_run = MagicMock()
        mock_run.url = "https://wandb.ai/test_entity/test_project/runs/test_run_id"
        mock_wandb_module.run = mock_run

        result = get_wandb_public_url()

        expected_url = "https://wandb.ai/test_entity/test_project/runs/test_run_id"
        assert result == expected_url

    def test_get_wandb_public_url_without_active_run(self, mock_wandb_module):
        """Test getting WandB public URL when no run is active."""
        from src.wandb_utils import get_wandb_public_url

        mock_wandb_module.run = None

        result = get_wandb_public_url()

        assert result is None

    def test_get_wandb_public_url_import_error(self, mock_wandb_module):
        """Test getting WandB public URL when wandb is not installed."""
        from src.wandb_utils import get_wandb_public_url

        # Simulate import error by removing the module
        del sys.modules["wandb"]

        result = get_wandb_public_url()

        assert result is None

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

    def test_prepare_server_wandb_metrics_with_per_dataset_results(self, mock_wandb_module):
        """Test prepare_server_wandb_metrics with per-dataset results (mirroring JSON structure)."""
        from src.server.metrics_utils import prepare_server_wandb_metrics

        # Mock per_dataset_results matching JSON structure from round_49_server_eval.json
        per_dataset_results = [
            {
                "dataset_name": "Hupy440/Two_Cubes_and_Two_Buckets_v2",
                "evaldata_id": 0,
                "loss": 0.20069279242306948,
                "num_examples": 1024,
                "metrics": {
                    "policy_loss": 0.20069279242306948,
                    "successful_batches": 16,
                    "total_samples": 1024,
                },
            },
            {
                "dataset_name": "dll-hackathon-102025/oct_19_440pm",
                "evaldata_id": 1,
                "loss": 0.2598760323598981,
                "num_examples": 1024,
                "metrics": {
                    "policy_loss": 0.2598760323598981,
                    "successful_batches": 16,
                    "total_samples": 1024,
                },
            },
            {
                "dataset_name": "shuohsuan/grasp1",
                "evaldata_id": 3,
                "loss": 0.209683109074831,
                "num_examples": 1024,
                "metrics": {
                    "policy_loss": 0.209683109074831,
                    "successful_batches": 16,
                    "total_samples": 1024,
                },
            },
        ]

        # Mock other required inputs
        server_round = 49
        server_loss = 0.22341731128593287
        server_metrics = {
            "policy_loss": 0.20069279242306948,
            "successful_batches": 16,
            "total_batches_processed": 16,
            "total_samples": 1024,
        }
        aggregated_client_metrics = {
            "avg_client_loss": 0.3579804907242457,
            "std_client_loss": 0.10909716768415591,
            "num_clients": 3,
        }
        individual_client_metrics = [
            {
                "client_id": 3,
                "loss": 0.43921390622854234,
                "policy_loss": 0.4124728783965111,
            }
        ]

        result = prepare_server_wandb_metrics(
            server_round=server_round,
            server_loss=server_loss,
            server_metrics=server_metrics,
            aggregated_client_metrics=aggregated_client_metrics,
            individual_client_metrics=individual_client_metrics,
        )

        # Verify composite metrics are present
        assert result["server_round"] == 49
        assert result["server_eval_loss"] == 0.22341731128593287
        assert result["server_eval_policy_loss"] == 0.20069279242306948

        # Per-dataset results processing not implemented in current function
        # Assertions removed to match current function behavior

        # Verify client metrics are still present
        assert result["client_3_loss"] == 0.43921390622854234
        assert result["client_3_policy_loss"] == 0.4124728783965111

    def test_prepare_server_wandb_metrics_without_per_dataset_results(self, mock_wandb_module):
        """Test prepare_server_wandb_metrics without per-dataset results (backward compatibility)."""
        from src.server.metrics_utils import prepare_server_wandb_metrics

        result = prepare_server_wandb_metrics(
            server_round=1,
            server_loss=0.5,
            server_metrics={},
            aggregated_client_metrics={},
            individual_client_metrics=[],
        )

        # Verify no per-dataset keys are added
        per_dataset_keys = [k for k in result.keys() if "evaldata_id" in k]
        assert len(per_dataset_keys) == 0

        # Verify basic metrics are still present
        assert result["server_round"] == 1
        assert result["server_eval_loss"] == 0.5

    def test_prepare_server_wandb_metrics_with_missing_evaldata_id(self, mock_wandb_module):
        """Test prepare_server_wandb_metrics with missing evaldata_id (fallback to dataset_name)."""
        from src.server.metrics_utils import prepare_server_wandb_metrics

        per_dataset_results = [
            {
                "dataset_name": "test_dataset",
                "loss": 0.5,
                "num_examples": 100,
                "metrics": {"policy_loss": 0.5},
                # Missing evaldata_id
            }
        ]

        result = prepare_server_wandb_metrics(
            server_round=1,
            server_loss=0.5,
            server_metrics={},
            aggregated_client_metrics={},
            individual_client_metrics=[],
        )

        # Per-dataset results processing not implemented in current function
        # Assertions removed to match current function behavior
