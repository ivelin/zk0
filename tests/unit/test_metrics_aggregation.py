"""Unit tests for client metrics aggregation functions."""

from src.server.metrics_utils import (
    aggregate_client_metrics,
    collect_individual_client_metrics,
)


def test_aggregate_client_metrics_empty():
    """Test computing aggregated metrics with no results."""
    result = aggregate_client_metrics([])
    expected = {
        "avg_client_loss": 0.0,
        "std_client_loss": 0.0,
        "avg_client_proximal_loss": 0.0,
        "avg_client_grad_norm": 0.0,
        "num_clients": 0,
    }
    assert result == expected


def test_aggregate_client_metrics_single_client():
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

    result = aggregate_client_metrics(validated_results)

    assert result["avg_client_loss"] == 1.5
    assert result["std_client_loss"] == 0.0  # Single client, no std
    assert result["avg_client_proximal_loss"] == 0.2
    assert result["avg_client_grad_norm"] == 0.8
    assert result["num_clients"] == 1


def test_aggregate_client_metrics_multiple_clients():
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

    result = aggregate_client_metrics(validated_results)

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


def test_collect_individual_client_metrics_empty():
    """Test collecting client metrics with no results."""
    result = collect_individual_client_metrics([])
    assert result == []


def test_collect_individual_client_metrics_single_client():
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

    result = collect_individual_client_metrics(validated_results)

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


def test_collect_individual_client_metrics_missing_fields():
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

    result = collect_individual_client_metrics(validated_results)

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