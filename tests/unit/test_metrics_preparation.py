"""Unit tests for in-memory metrics preparation functions."""

from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import numpy as np


class TestPrepareCheckpointMetrics:
    """Test prepare_server_eval_metrics helper function."""

    def test_prepare_checkpoint_metrics_populated(self):
        """Test metrics preparation with populated strategy data."""
        from src.server.server_utils import prepare_server_eval_metrics

        mock_strategy = MagicMock()
        mock_strategy.server_eval_losses = [0.723, 0.655]
        mock_strategy.last_aggregated_metrics = {"avg_client_loss": 0.894, "num_clients": 2}
        mock_strategy.last_client_metrics = [{"client_id": 6}, {"client_id": 2}]
        mock_strategy.last_per_dataset_results = [{"loss": 0.578, "dataset_name": "Hupy440/Two_Cubes_and_Two_Buckets_v2"}]

        metrics = prepare_server_eval_metrics(mock_strategy, server_round=0)

        assert metrics["composite_eval_loss"] == 0.655
        assert metrics["aggregated_client_metrics"]["avg_client_loss"] == 0.894
        assert len(metrics["individual_client_metrics"]) == 2
        assert len(metrics["server_eval_dataset_results"]) == 1

    def test_prepare_checkpoint_metrics_empty(self):
        """Test metrics preparation with empty strategy data."""
        from src.server.server_utils import prepare_server_eval_metrics

        mock_strategy = MagicMock()
        mock_strategy.server_eval_losses = []
        mock_strategy.last_aggregated_metrics = {}
        mock_strategy.last_client_metrics = []
        mock_strategy.last_per_dataset_results = []

        metrics = prepare_server_eval_metrics(mock_strategy, server_round=0)

        assert metrics["composite_eval_loss"] == "N/A"
        assert metrics["aggregated_client_metrics"] == {}
        assert len(metrics["individual_client_metrics"]) == 0
        assert len(metrics["server_eval_dataset_results"]) == 0


class TestInMemoryExtractFinalMetrics:
    """Test in-memory final metrics extraction in save_model_checkpoint."""

    def test_in_memory_metrics_populated(self):
        """Test in-memory metrics extraction with populated strategy."""
        from src.server.server_utils import save_model_checkpoint

        mock_strategy = MagicMock()
        mock_strategy.server_eval_losses = [0.655]
        mock_strategy.last_aggregated_metrics = {"avg_client_loss": 0.894, "num_clients": 2}
        mock_strategy.last_client_metrics = [{"client_id": 6, "policy_loss": 0.778}]
        mock_strategy.last_per_dataset_results = [{"loss": 0.578, "dataset_name": "test_dataset"}]
        mock_strategy.template_model = MagicMock()
        mock_strategy.models_dir = Path(tempfile.mkdtemp())
        mock_strategy.server_dir = Path(tempfile.mkdtemp())

        mock_parameters = MagicMock()
        with patch("flwr.common.parameters_to_ndarrays") as mock_ndarrays, \
              patch("src.server.server_utils.generate_model_card") as mock_generate, \
              patch("src.server.model_utils.extract_final_metrics") as mock_extract_metrics:
            mock_ndarrays.return_value = [np.array([1.0])]
            mock_generate.return_value = "# Model Card"
            mock_extract_metrics.return_value = {
                "composite_eval_loss": 0.655,
                "aggregated_client_metrics": {"avg_client_loss": 0.894},
                "individual_client_metrics": [{"client_id": 6}],
                "server_eval_dataset_results": [{"loss": 0.578}]
            }

            # Call save_model_checkpoint (which uses in-memory extraction)
            save_model_checkpoint(mock_strategy, mock_parameters, 2)

            # Verify generate_model_card was called with correct metrics
            mock_generate.assert_called_once()
            args = mock_generate.call_args[0]
            metrics_arg = args[3]  # 4th arg is metrics
            assert metrics_arg["composite_eval_loss"] == 0.655
            assert len(metrics_arg["server_eval_dataset_results"]) == 1


class TestInMemoryExtractTrainingInsights:
    """Test in-memory training insights computation."""

    def test_in_memory_insights_populated(self):
        """Test in-memory insights with populated history."""
        from src.server.server_utils import compute_in_memory_insights  # Assuming helper is extracted

        mock_strategy = MagicMock()
        mock_strategy.federated_metrics_history = [
            {"round": 1, "avg_client_loss": 1.0, "num_clients": 2, "param_update_norm": 0.01},
            {"round": 2, "avg_client_loss": 0.8, "num_clients": 2, "param_update_norm": 0.008},
        ]
        mock_strategy.server_eval_losses = [0.9, 0.5]

        insights = compute_in_memory_insights(mock_strategy)

        assert "Started at 1.0000, ended at 0.8000" in insights["avg_client_loss_trend"]
        assert "0.9000 → 0.5000" in insights["convergence_trend"]
        assert "Average 2.0 clients" in insights["client_participation_rate"]
        assert insights["anomalies"] == []  # No dropouts

    def test_in_memory_insights_with_anomalies(self):
        """Test in-memory insights with client dropouts."""
        from src.server.server_utils import compute_in_memory_insights

        mock_strategy = MagicMock()
        mock_strategy.federated_metrics_history = [
            {"round": 1, "avg_client_loss": 1.0, "num_clients": 2, "param_update_norm": 0.01},
            {"round": 2, "avg_client_loss": 0.8, "num_clients": 1, "param_update_norm": 0.008},  # Dropout
        ]
        mock_strategy.server_eval_losses = [0.9, 0.5]

        insights = compute_in_memory_insights(mock_strategy)

        assert "Client dropouts in rounds: [2]" in insights["anomalies"][0]

    def test_in_memory_insights_empty(self):
        """Test in-memory insights with empty history."""
        from src.server.server_utils import compute_in_memory_insights

        mock_strategy = MagicMock()
        mock_strategy.federated_metrics_history = []
        mock_strategy.server_eval_losses = []

        insights = compute_in_memory_insights(mock_strategy)

        assert insights["convergence_trend"] == "N/A"
        assert insights["avg_client_loss_trend"] == "N/A"
        assert insights["client_participation_rate"] == "N/A"
        assert insights["anomalies"] == []