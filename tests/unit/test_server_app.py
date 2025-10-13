"""Unit tests for server_app.py."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from src.server_app import compute_server_param_update_norm, aggregate_client_metrics
from flwr.common import ndarrays_to_parameters


class TestComputeServerParamUpdateNorm:
    """Test cases for compute_server_param_update_norm function."""

    def test_compute_server_param_update_norm_with_changes(self):
        """Test parameter update norm calculation with actual parameter changes."""
        # Create test parameter arrays
        prev_arrays = [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[0.1]], dtype=np.float32),
        ]

        curr_arrays = [
            np.array([[1.1, 2.1]], dtype=np.float32),  # Changed by 0.1
            np.array([[0.2]], dtype=np.float32),      # Changed by 0.1
        ]

        # Convert to Flower Parameters
        prev_params = ndarrays_to_parameters(prev_arrays)
        curr_params = ndarrays_to_parameters(curr_arrays)

        # Calculate expected norm: sqrt(sum of squared differences)
        # Differences: [[0.1, 0.1]] and [[0.1]] = sum of squares: 0.04
        # Norm: sqrt(0.04) ≈ 0.2
        expected_norm = np.sqrt(sum(np.sum((c - p)**2) for c, p in zip(curr_arrays, prev_arrays)))

        result = compute_server_param_update_norm(prev_params, curr_params)
        assert abs(result - expected_norm) < 1e-6
        assert result > 0.0

    def test_compute_server_param_update_norm_identical_params(self):
        """Test parameter update norm with identical parameters."""
        arrays = [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[0.1]], dtype=np.float32),
        ]

        params1 = ndarrays_to_parameters(arrays)
        params2 = ndarrays_to_parameters(arrays)  # Identical

        result = compute_server_param_update_norm(params1, params2)
        assert result == 0.0

    def test_compute_server_param_update_norm_none_params(self):
        """Test parameter update norm with None parameters."""
        arrays = [np.array([[1.0]], dtype=np.float32)]
        params = ndarrays_to_parameters(arrays)

        result = compute_server_param_update_norm(None, params)
        assert result == 0.0

        result = compute_server_param_update_norm(params, None)
        assert result == 0.0

    def test_compute_server_param_update_norm_mismatched_lengths(self):
        """Test parameter update norm with mismatched parameter list lengths."""
        arrays1 = [np.array([[1.0]], dtype=np.float32)]
        arrays2 = [np.array([[1.0]], dtype=np.float32), np.array([[2.0]], dtype=np.float32)]

        params1 = ndarrays_to_parameters(arrays1)
        params2 = ndarrays_to_parameters(arrays2)

        result = compute_server_param_update_norm(params1, params2)
        assert result == 0.0


class TestAggregateClientMetrics:
    """Test cases for aggregate_client_metrics function."""

    def test_aggregate_client_metrics_with_valid_results(self):
        """Test client metrics aggregation with valid fit results."""
        # Create mock fit results
        mock_fit_res1 = MagicMock()
        mock_fit_res1.metrics = {
            "loss": 0.5,
            "fedprox_loss": 0.01,
            "grad_norm": 1.0,
        }

        mock_fit_res2 = MagicMock()
        mock_fit_res2.metrics = {
            "loss": 0.7,
            "fedprox_loss": 0.02,
            "grad_norm": 1.5,
        }

        validated_results = [
            (MagicMock(), mock_fit_res1),
            (MagicMock(), mock_fit_res2),
        ]

        result = aggregate_client_metrics(validated_results)

        # Check aggregated values
        assert result["num_clients"] == 2
        assert abs(result["avg_client_loss"] - 0.6) < 1e-6  # (0.5 + 0.7) / 2
        assert abs(result["std_client_loss"] - 0.1) < 1e-6  # std([0.5, 0.7])
        assert abs(result["avg_client_proximal_loss"] - 0.015) < 1e-6  # (0.01 + 0.02) / 2
        assert abs(result["avg_client_grad_norm"] - 1.25) < 1e-6  # (1.0 + 1.5) / 2

    def test_aggregate_client_metrics_with_missing_metrics(self):
        """Test client metrics aggregation when some metrics are missing."""
        # Create mock fit results with missing metrics
        mock_fit_res1 = MagicMock()
        mock_fit_res1.metrics = {"loss": 0.5}  # Missing fedprox_loss and grad_norm

        mock_fit_res2 = MagicMock()
        mock_fit_res2.metrics = {"loss": 0.7, "fedprox_loss": 0.02}  # Missing grad_norm

        validated_results = [
            (MagicMock(), mock_fit_res1),
            (MagicMock(), mock_fit_res2),
        ]

        result = aggregate_client_metrics(validated_results)

        # Check that missing metrics default to 0.0
        assert result["num_clients"] == 2
        assert abs(result["avg_client_loss"] - 0.6) < 1e-6
        assert abs(result["avg_client_proximal_loss"] - 0.01) < 1e-6  # (0.0 + 0.02) / 2
        assert result["avg_client_grad_norm"] == 0.0  # Both missing

    def test_aggregate_client_metrics_empty_results(self):
        """Test client metrics aggregation with empty results."""
        result = aggregate_client_metrics([])

        # Check default values
        assert result["num_clients"] == 0
        assert result["avg_client_loss"] == 0.0
        assert result["std_client_loss"] == 0.0
        assert result["avg_client_proximal_loss"] == 0.0
        assert result["avg_client_grad_norm"] == 0.0

    def test_aggregate_client_metrics_single_client(self):
        """Test client metrics aggregation with single client (no std calculation)."""
        mock_fit_res = MagicMock()
        mock_fit_res.metrics = {
            "loss": 0.5,
            "fedprox_loss": 0.01,
            "grad_norm": 1.0,
        }

        validated_results = [(MagicMock(), mock_fit_res)]

        result = aggregate_client_metrics(validated_results)

        assert result["num_clients"] == 1
        assert result["avg_client_loss"] == 0.5
        assert result["std_client_loss"] == 0.0  # No std for single value
        assert result["avg_client_proximal_loss"] == 0.01
        assert result["avg_client_grad_norm"] == 1.0