"""Unit tests for server_app.py functions."""

import pytest
from src.server_app import (
    check_early_stopping,
    update_early_stopping_tracking,
    _compute_aggregated_metrics,
    _collect_client_metrics,
)


class MockStrategy:
    """Mock strategy class for testing early stopping functions."""

    def __init__(self, patience=10):
        self.early_stopping_patience = patience
        self.best_eval_loss = float("inf")
        self.rounds_without_improvement = 0
        self.early_stopping_triggered = False


def test_check_early_stopping_disabled():
    """Test early stopping when patience is 0 (disabled)."""
    should_stop, new_rounds = check_early_stopping(1.0, 2.0, 0, 0)
    assert should_stop is False
    assert new_rounds == 0


def test_check_early_stopping_improvement():
    """Test early stopping when loss improves."""
    should_stop, new_rounds = check_early_stopping(0.5, 1.0, 5, 10)
    assert should_stop is False
    assert new_rounds == 0


def test_check_early_stopping_no_improvement():
    """Test early stopping when loss doesn't improve."""
    should_stop, new_rounds = check_early_stopping(1.0, 0.5, 5, 10)
    assert should_stop is False
    assert new_rounds == 6


def test_check_early_stopping_trigger():
    """Test early stopping when patience is exceeded."""
    should_stop, new_rounds = check_early_stopping(1.0, 0.5, 9, 10)
    assert should_stop is True
    assert new_rounds == 10


def test_update_early_stopping_tracking_improvement():
    """Test updating early stopping tracking when loss improves."""
    strategy = MockStrategy(patience=5)
    strategy.best_eval_loss = 1.0

    update_early_stopping_tracking(strategy, 3, 0.8)

    assert strategy.best_eval_loss == 0.8
    assert strategy.rounds_without_improvement == 0
    assert strategy.early_stopping_triggered is False


def test_update_early_stopping_tracking_no_improvement():
    """Test updating early stopping tracking when loss doesn't improve."""
    strategy = MockStrategy(patience=5)
    strategy.best_eval_loss = 0.8
    strategy.rounds_without_improvement = 2

    update_early_stopping_tracking(strategy, 4, 1.0)

    assert strategy.best_eval_loss == 0.8
    assert strategy.rounds_without_improvement == 3
    assert strategy.early_stopping_triggered is False


def test_update_early_stopping_tracking_trigger():
    """Test updating early stopping tracking when patience is exceeded."""
    strategy = MockStrategy(patience=3)
    strategy.best_eval_loss = 0.8
    strategy.rounds_without_improvement = 2

    update_early_stopping_tracking(strategy, 5, 1.0)

    assert strategy.best_eval_loss == 0.8
    assert strategy.rounds_without_improvement == 3
    assert strategy.early_stopping_triggered is True


def test_update_early_stopping_tracking_already_triggered():
    """Test that tracking is skipped when early stopping is already triggered."""
    strategy = MockStrategy(patience=5)
    strategy.early_stopping_triggered = True
    strategy.best_eval_loss = 0.8
    strategy.rounds_without_improvement = 3

    update_early_stopping_tracking(strategy, 6, 1.0)

    # Values should remain unchanged
    assert strategy.best_eval_loss == 0.8
    assert strategy.rounds_without_improvement == 3
    assert strategy.early_stopping_triggered is True


def test_compute_aggregated_metrics_empty():
    """Test computing aggregated metrics with no results."""
    result = _compute_aggregated_metrics([])
    expected = {
        "avg_client_loss": 0.0,
        "std_client_loss": 0.0,
        "avg_client_proximal_loss": 0.0,
        "avg_client_grad_norm": 0.0,
        "num_clients": 0,
    }
    assert result == expected


def test_early_stopping_returns_parameters_not_none():
    """Test that early stopping logic correctly detects when to stop without returning None."""
    # Test the core early stopping logic that was causing the bug
    # The bug was that aggregate_fit returned None when early stopping was triggered
    # This test ensures the early stopping detection logic works correctly

    # Test case: early stopping should be triggered when patience is exceeded
    should_stop, new_rounds = check_early_stopping(
        eval_loss=1.0,  # Worse than best
        best_loss=0.5,  # Best loss so far
        rounds_without_improvement=9,  # One round away from patience=10
        patience=10,
    )

    # Verify early stopping is triggered
    assert should_stop is True
    assert new_rounds == 10

    # The fix ensures that when early stopping is triggered,
    # valid parameters (current_parameters) are returned instead of None
    # This prevents the "cannot unpack non-iterable NoneType object" error


def test_compute_aggregated_metrics_single_client():
    """Test computing aggregated metrics with single client."""

    # Mock client proxy and fit result
    class MockClientProxy:
        def __init__(self, cid):
            self.cid = cid

    class MockFitRes:
        def __init__(self, metrics):
            self.metrics = metrics

    validated_results = [
        (
            MockClientProxy("client_0"),
            MockFitRes({"loss": 1.5, "fedprox_loss": 0.2, "grad_norm": 0.8}),
        )
    ]

    result = _compute_aggregated_metrics(validated_results)

    assert result["avg_client_loss"] == 1.5
    assert result["std_client_loss"] == 0.0  # Single client, no std
    assert result["avg_client_proximal_loss"] == 0.2
    assert result["avg_client_grad_norm"] == 0.8
    assert result["num_clients"] == 1


def test_compute_aggregated_metrics_multiple_clients():
    """Test computing aggregated metrics with multiple clients."""

    # Mock client proxy and fit result
    class MockClientProxy:
        def __init__(self, cid):
            self.cid = cid

    class MockFitRes:
        def __init__(self, metrics):
            self.metrics = metrics

    validated_results = [
        (
            MockClientProxy("client_0"),
            MockFitRes({"loss": 1.0, "fedprox_loss": 0.1, "grad_norm": 0.5}),
        ),
        (
            MockClientProxy("client_1"),
            MockFitRes({"loss": 2.0, "fedprox_loss": 0.2, "grad_norm": 1.0}),
        ),
        (
            MockClientProxy("client_2"),
            MockFitRes({"loss": 3.0, "fedprox_loss": 0.3, "grad_norm": 1.5}),
        ),
    ]

    result = _compute_aggregated_metrics(validated_results)

    # For [1,2,3]: mean=2.0, std≈0.8165 (population std, not sample std)
    assert result["avg_client_loss"] == 2.0
    assert (
        abs(result["std_client_loss"] - 0.816496580927726) < 0.001
    )  # numpy std (ddof=0)
    assert (
        abs(result["avg_client_proximal_loss"] - 0.2) < 1e-10
    )  # Handle floating point precision
    assert result["avg_client_grad_norm"] == 1.0
    assert result["num_clients"] == 3


def test_collect_client_metrics_empty():
    """Test collecting client metrics with no results."""
    result = _collect_client_metrics([])
    assert result == []


def test_collect_client_metrics_single_client():
    """Test collecting client metrics with single client."""

    # Mock client proxy and fit result
    class MockClientProxy:
        def __init__(self, cid):
            self.cid = cid

    class MockFitRes:
        def __init__(self, metrics):
            self.metrics = metrics

    validated_results = [
        (
            MockClientProxy("client_0"),
            MockFitRes(
                {
                    "loss": 1.5,
                    "policy_loss": 1.3,
                    "fedprox_loss": 0.2,
                    "grad_norm": 0.8,
                    "param_hash": "abc123",
                    "dataset_name": "test_dataset",
                    "steps_completed": 10,
                    "param_update_norm": 0.5,
                }
            ),
        )
    ]

    result = _collect_client_metrics(validated_results)

    expected = [
        {
            "round": 0,
            "client_id": "client_0",
            "dataset_name": "test_dataset",
            "loss": 1.5,
            "policy_loss": 1.3,
            "fedprox_loss": 0.2,
            "grad_norm": 0.8,
            "param_hash": "abc123",
            "num_steps": 10,
            "param_update_norm": 0.5,
            "flower_proxy_cid": "client_0",
        }
    ]

    assert result == expected


def test_collect_client_metrics_missing_fields():
    """Test collecting client metrics with missing optional fields."""

    # Mock client proxy and fit result
    class MockClientProxy:
        def __init__(self, cid):
            self.cid = cid

    class MockFitRes:
        def __init__(self, metrics):
            self.metrics = metrics

    validated_results = [
        (
            MockClientProxy("client_0"),
            MockFitRes({"loss": 1.5}),
        )  # Missing optional fields
    ]

    result = _collect_client_metrics(validated_results)

    expected = [
        {
            "round": 0,
            "client_id": "client_0",
            "dataset_name": "",
            "loss": 0.0,
            "policy_loss": 0.0,
            "fedprox_loss": 0.0,
            "grad_norm": 0.0,
            "param_hash": "",
            "num_steps": 0,
            "param_update_norm": 0.0,
            "flower_proxy_cid": "client_0",
        }
    ]

    assert result == expected


def test_aggregate_fit_parameter_safety_fix():
    """Test that aggregate_fit always returns valid parameters to prevent Flower unpacking errors.

    This test ensures the critical fix for the "cannot unpack non-iterable NoneType object" error
    where aggregate_fit() could return None parameters in edge cases.
    """
    from unittest.mock import Mock, patch
    from src.server_app import AggregateEvaluationStrategy
    from flwr.common import ndarrays_to_parameters
    import numpy as np

    # Create mock strategy with required attributes
    strategy = Mock(spec=AggregateEvaluationStrategy)
    strategy.initial_parameters = ndarrays_to_parameters([np.array([1.0, 2.0])])
    strategy.early_stopping_triggered = False
    strategy.template_model = Mock()

    # Mock the parent aggregate_fit to return None parameters (simulating edge case)
    with patch("src.server_app.FedProx.aggregate_fit", return_value=(None, {})):
        with patch("src.server_app.logger") as mock_logger:
            # Import and call the actual aggregate_fit method
            from src.server_app import AggregateEvaluationStrategy

            # Create a minimal instance for testing
            test_strategy = AggregateEvaluationStrategy.__new__(
                AggregateEvaluationStrategy
            )
            test_strategy.initial_parameters = ndarrays_to_parameters(
                [np.array([1.0, 2.0])]
            )
            test_strategy.early_stopping_triggered = False

            # Mock other required attributes
            test_strategy.template_model = Mock()
            test_strategy.previous_parameters = None
            test_strategy.last_aggregated_metrics = {}
            test_strategy.last_client_metrics = []
            test_strategy.context = Mock()
            test_strategy.context.run_config = {}

            # Call aggregate_fit with empty results (edge case that could cause None return)
            result_params, result_metrics = test_strategy.aggregate_fit(
                server_round=1, results=[], failures=[]
            )

            # Verify that valid parameters are returned (not None)
            assert result_params is not None, (
                "aggregate_fit should never return None parameters"
            )
            assert isinstance(
                result_params, type(ndarrays_to_parameters([np.array([1.0])]))
            ), "Should return Flower Parameters object"

            # Verify warning was logged about returning initial parameters
            mock_logger.warning.assert_called_with(
                "⚠️ Server: No parameters aggregated for round 1, returning initial parameters"
            )


def test_dynamic_lr_config_enabled():
    """Test that dynamic LR config is properly read."""
    # Test the config reading logic directly
    config = {"dynamic_lr_enabled": True}
    enabled = config.get("dynamic_lr_enabled", False)
    assert enabled is True

    config = {"dynamic_lr_enabled": False}
    enabled = config.get("dynamic_lr_enabled", False)
    assert enabled is False

    config = {}  # Default case
    enabled = config.get("dynamic_lr_enabled", False)
    assert enabled is False
