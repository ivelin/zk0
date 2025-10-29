"""Unit tests for file-based metrics and insights extraction functions."""

import pytest
import json
import tempfile
from pathlib import Path


class TestExtractFinalMetrics:
    """Test extract_final_metrics function."""

    def test_extract_final_metrics_success(self):
        """Test successful metrics extraction."""
        from src.server.server_utils import extract_final_metrics

        with tempfile.TemporaryDirectory() as temp_dir:
            server_dir = Path(temp_dir)
            metrics_file = server_dir / "round_50_server_eval.json"

            test_data = {
                "loss": 0.456,
                "aggregated_client_metrics": {"avg_client_loss": 0.7},
                "individual_client_metrics": [{"client_id": 0}, {"client_id": 1}],
                "metrics": {"per_dataset_results": [{"loss": 0.4}]}
            }

            with open(metrics_file, "w") as f:
                json.dump(test_data, f)

            metrics = extract_final_metrics(server_dir, 50)

            assert metrics["composite_eval_loss"] == 0.456
            assert metrics["aggregated_client_metrics"]["avg_client_loss"] == 0.7
            assert len(metrics["individual_client_metrics"]) == 2

    def test_extract_final_metrics_missing_file(self):
        """Test metrics extraction when file doesn't exist."""
        from src.server.server_utils import extract_final_metrics

        with tempfile.TemporaryDirectory() as temp_dir:
            server_dir = Path(temp_dir)

            metrics = extract_final_metrics(server_dir, 50)

            assert metrics["composite_eval_loss"] == "N/A"
            assert metrics["aggregated_client_metrics"] == {}
            assert metrics["individual_client_metrics"] == []


class TestExtractTrainingInsights:
    """Test extract_training_insights function."""

    def test_extract_training_insights_success(self):
        """Test successful insights extraction."""
        from src.server.server_utils import extract_training_insights

        with tempfile.TemporaryDirectory() as temp_dir:
            server_dir = Path(temp_dir)

            # Create federated_metrics.json
            metrics_data = [
                {"round": 1, "avg_client_loss": 1.0, "param_update_norm": 0.01, "num_clients": 2},
                {"round": 2, "avg_client_loss": 0.8, "param_update_norm": 0.008, "num_clients": 2},
            ]
            with open(server_dir / "federated_metrics.json", "w") as f:
                json.dump(metrics_data, f)

            # Create policy_loss_history.json
            history_data = {
                "1": {"server_policy_loss": 0.9},
                "2": {"server_policy_loss": 0.5},
            }
            with open(server_dir / "policy_loss_history.json", "w") as f:
                json.dump(history_data, f)

            insights = extract_training_insights(server_dir, 2)

            assert "Started at 1.0000, ended at 0.8000" in insights["avg_client_loss_trend"]
            assert "0.9000 → 0.5000" in insights["convergence_trend"]
            assert "Average 2.0 clients" in insights["client_participation_rate"]

    def test_extract_training_insights_missing_files(self):
        """Test insights extraction when files don't exist."""
        from src.server.server_utils import extract_training_insights

        with tempfile.TemporaryDirectory() as temp_dir:
            server_dir = Path(temp_dir)

            insights = extract_training_insights(server_dir, 2)

            assert insights["convergence_trend"] == "N/A"
            assert insights["avg_client_loss_trend"] == "N/A"
            assert insights["anomalies"] == []