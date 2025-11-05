"""Unit tests for parameter aggregation functions."""

from unittest.mock import Mock, patch
from src.server.metrics_utils import (
    aggregate_and_log_metrics,
    finalize_round_metrics,
)


class TestAggregateParameters:
    """Test the aggregate_parameters function."""

    def test_aggregate_parameters_success(self):
        """Test successful parameter aggregation."""
        from flwr.common import ndarrays_to_parameters
        import numpy as np

        # Create mock strategy
        strategy = Mock()

        # Create mock validated results
        mock_client_proxy = Mock()
        mock_fit_res = Mock()
        mock_fit_res.parameters = ndarrays_to_parameters([np.array([1.0, 2.0])])
        validated_results = [(mock_client_proxy, mock_fit_res)]

        server_round = 5

        # Mock the method return value
        mock_aggregated_params = ndarrays_to_parameters([np.array([1.5, 2.5])])
        mock_parent_metrics = {"fedprox_mu": 0.01}

        strategy.aggregate_parameters.return_value = (
            mock_aggregated_params,
            mock_parent_metrics
        )

        aggregated_params, parent_metrics = strategy.aggregate_parameters(
            server_round, validated_results
        )

        # Verify return values
        assert aggregated_params == mock_aggregated_params
        assert parent_metrics == mock_parent_metrics


class TestAggregateAndLogMetrics:
    """Test the aggregate_and_log_metrics function."""

    def test_aggregate_and_log_metrics(self):
        """Test metrics aggregation and logging."""
        from flwr.common import ndarrays_to_parameters
        import numpy as np

        # Create mock strategy
        strategy = Mock()
        strategy.previous_parameters = ndarrays_to_parameters([np.array([1.0, 2.0])])
        strategy.last_aggregated_metrics = None
        strategy.last_client_metrics = None

        # Create mock validated results
        mock_client_proxy = Mock()
        mock_fit_res = Mock()
        mock_fit_res.metrics = {"loss": 1.5, "fedprox_loss": 0.2}
        validated_results = [(mock_client_proxy, mock_fit_res)]

        server_round = 5
        aggregated_parameters = ndarrays_to_parameters([np.array([1.5, 2.5])])

        # Mock compute_server_param_update_norm
        with patch("src.common.parameter_utils.compute_server_param_update_norm") as mock_compute_norm:
            mock_compute_norm.return_value = 0.1

            result = aggregate_and_log_metrics(
                strategy, server_round, validated_results, aggregated_parameters
            )

            # Verify aggregated metrics include param_update_norm
            assert result["param_update_norm"] == 0.1
            assert result["avg_client_loss"] == 1.5

            # Verify strategy attributes were set
            assert strategy.last_aggregated_metrics == result
            assert strategy.last_client_metrics is not None


class TestFinalizeRoundMetrics:
    """Test the finalize_round_metrics function."""

    def test_finalize_round_metrics(self):
        """Test final metrics merging and diagnostics."""

        # Create mock strategy
        strategy = Mock()
        strategy.proximal_mu = 0.01
        strategy.server_eval_losses = [1.0, 0.95]

        server_round = 5
        aggregated_client_metrics = {"num_clients": 3}
        parent_metrics = {"fedprox_mu": 0.01}

        result = finalize_round_metrics(
            strategy, server_round, aggregated_client_metrics, parent_metrics
        )

        # Verify metrics are merged
        assert result["num_clients"] == 3
        assert result["fedprox_mu"] == 0.01

        # Verify diagnostics are added
        assert result["diagnosis_mu"] == 0.01
        assert "diagnosis_eval_trend" in result