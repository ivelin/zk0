"""Unit tests for prepare_server_eval_metrics function."""

from unittest.mock import Mock
from src.server.server_utils import prepare_server_eval_metrics


class TestPrepareServerEvalMetrics:
    """Test the prepare_server_eval_metrics function."""

    def test_prepare_server_eval_metrics_complete_data(self):
        """Test preparing server eval metrics with complete data."""

        # Mock strategy with complete data
        strategy = Mock()
        strategy.server_eval_losses = [1.0, 0.95, 0.9]
        strategy.last_aggregated_metrics = {
            "avg_client_loss": 1.2,
            "std_client_loss": 0.3,
            "num_clients": 3,
            "param_update_norm": 0.5,
        }
        strategy.last_client_metrics = [
            {
                "round": 0,
                "client_id": "client_0",
                "dataset_name": "test_dataset",
                "loss": 1.0,
                "policy_loss": 0.8,
                "fedprox_loss": 0.2,
                "grad_norm": 0.5,
                "param_hash": "abc123",
                "num_steps": 10,
                "param_update_norm": 0.1,
                "flower_proxy_cid": "cid_0",
            }
        ]
        strategy.last_per_dataset_results = [
            {
                "dataset_name": "dataset1",
                "evaldata_id": 123,
                "loss": 0.8,
                "num_examples": 100,
                "metrics": {
                    "policy_loss": 0.8,
                    "action_dim": 7,
                    "successful_batches": 2,
                },
            },
            {
                "dataset_name": "dataset2",
                "evaldata_id": 456,
                "loss": 1.2,
                "num_examples": 150,
                "metrics": {
                    "policy_loss": 1.2,
                    "action_dim": 7,
                    "successful_batches": 3,
                },
            },
        ]

        server_round = 5

        result = prepare_server_eval_metrics(strategy, server_round)

        # Verify structure matches current implementation
        assert result["composite_eval_loss"] == 0.9  # Last eval loss
        assert result["num_datasets_evaluated"] == 2

        # Verify aggregated client metrics
        agg_metrics = result["aggregated_client_metrics"]
        assert agg_metrics["avg_client_loss"] == 1.2
        assert agg_metrics["std_client_loss"] == 0.3
        assert agg_metrics["num_clients"] == 3
        assert agg_metrics["param_update_norm"] == 0.5

        # Verify individual client metrics
        ind_metrics = result["individual_client_metrics"]
        assert len(ind_metrics) == 1
        assert ind_metrics[0]["client_id"] == "client_0"

        # Verify per-dataset results
        per_dataset = result["server_eval_dataset_results"]
        assert len(per_dataset) == 2
        assert per_dataset[0]["dataset_name"] == "dataset1"
        assert per_dataset[1]["dataset_name"] == "dataset2"

    def test_prepare_server_eval_metrics_minimal_data(self):
        """Test preparing server eval metrics with minimal data."""

        # Mock strategy with minimal data
        strategy = Mock()
        strategy.server_eval_losses = [0.5]
        strategy.last_aggregated_metrics = {}
        strategy.last_client_metrics = []
        strategy.last_per_dataset_results = []

        server_round = 1

        result = prepare_server_eval_metrics(strategy, server_round)

        # Verify structure matches current implementation
        assert result["composite_eval_loss"] == 0.5
        assert result["num_datasets_evaluated"] == 0

        # Verify empty collections
        assert result["aggregated_client_metrics"] == {}
        assert result["individual_client_metrics"] == []
        assert result["server_eval_dataset_results"] == []

    def test_prepare_server_eval_metrics_no_eval_losses(self):
        """Test preparing server eval metrics when no eval losses exist."""

        # Mock strategy without eval losses
        strategy = Mock()
        strategy.server_eval_losses = None
        strategy.last_aggregated_metrics = {"num_clients": 2}
        strategy.last_client_metrics = []
        strategy.last_per_dataset_results = []

        server_round = 3

        result = prepare_server_eval_metrics(strategy, server_round)

        # Should use default loss of "N/A"
        assert result["composite_eval_loss"] == "N/A"
        assert result["aggregated_client_metrics"]["num_clients"] == 2

    def test_prepare_server_eval_metrics_single_dataset(self):
        """Test preparing server eval metrics with single dataset."""

        # Mock strategy with single dataset
        strategy = Mock()
        strategy.server_eval_losses = [0.7]
        strategy.last_aggregated_metrics = {"avg_client_loss": 1.0}
        strategy.last_client_metrics = []
        strategy.last_per_dataset_results = [
            {
                "dataset_name": "single_dataset",
                "evaldata_id": None,  # No evaldata_id
                "loss": 0.7,
                "num_examples": 200,
                "metrics": {"policy_loss": 0.7},
            }
        ]

        server_round = 2

        result = prepare_server_eval_metrics(strategy, server_round)

        assert result["composite_eval_loss"] == 0.7
        assert result["num_datasets_evaluated"] == 1
        assert len(result["server_eval_dataset_results"]) == 1
        assert result["server_eval_dataset_results"][0]["dataset_name"] == "single_dataset"

    def test_prepare_server_eval_metrics_client_round_update(self):
        """Test that individual client metrics are preserved."""

        # Mock strategy with client metrics from previous round
        strategy = Mock()
        strategy.server_eval_losses = [0.6]
        strategy.last_aggregated_metrics = {}
        strategy.last_client_metrics = [
            {
                "round": 0,  # Old round
                "client_id": "client_1",
                "loss": 1.5,
            },
            {
                "round": 1,  # Different old round
                "client_id": "client_2",
                "loss": 2.0,
            }
        ]
        strategy.last_per_dataset_results = []

        server_round = 5

        result = prepare_server_eval_metrics(strategy, server_round)

        # Verify all client metrics are preserved
        ind_metrics = result["individual_client_metrics"]
        assert len(ind_metrics) == 2
        assert ind_metrics[0]["client_id"] == "client_1"
        assert ind_metrics[1]["client_id"] == "client_2"

    def test_prepare_server_eval_metrics_timestamp_format(self):
        """Test that function works without timestamp (not in current implementation)."""

        strategy = Mock()
        strategy.server_eval_losses = [0.5]
        strategy.last_aggregated_metrics = {}
        strategy.last_client_metrics = []
        strategy.last_per_dataset_results = []

        server_round = 1

        result = prepare_server_eval_metrics(strategy, server_round)

        # Current implementation doesn't include timestamp
        assert "timestamp" not in result
        assert result["composite_eval_loss"] == 0.5

    def test_prepare_server_eval_metrics_metrics_descriptions_complete(self):
        """Test that current implementation works without metrics_descriptions."""

        strategy = Mock()
        strategy.server_eval_losses = [0.5]
        strategy.last_aggregated_metrics = {}
        strategy.last_client_metrics = []
        strategy.last_per_dataset_results = []

        server_round = 1

        result = prepare_server_eval_metrics(strategy, server_round)

        # Current implementation doesn't include metrics_descriptions
        assert "metrics_descriptions" not in result
        assert result["composite_eval_loss"] == 0.5

    def test_prepare_server_eval_metrics_per_dataset_results_preserved(self):
        """Test that per-dataset results are properly preserved."""

        # Mock strategy with detailed per-dataset results
        strategy = Mock()
        strategy.server_eval_losses = [1.0]
        strategy.last_aggregated_metrics = {}
        strategy.last_client_metrics = []
        strategy.last_per_dataset_results = [
            {
                "dataset_name": "complex_dataset",
                "evaldata_id": 789,
                "loss": 1.0,
                "num_examples": 500,
                "metrics": {
                    "policy_loss": 1.0,
                    "action_dim": 7,
                    "successful_batches": 5,
                    "total_batches_processed": 5,
                    "total_samples": 500,
                    "custom_metric": "value"
                },
            }
        ]

        server_round = 10

        result = prepare_server_eval_metrics(strategy, server_round)

        # Verify per_dataset_results are preserved
        per_dataset = result["server_eval_dataset_results"]
        assert len(per_dataset) == 1
        assert per_dataset[0]["dataset_name"] == "complex_dataset"
        assert per_dataset[0]["evaldata_id"] == 789
        assert per_dataset[0]["loss"] == 1.0
        assert per_dataset[0]["num_examples"] == 500
        assert per_dataset[0]["metrics"]["custom_metric"] == "value"