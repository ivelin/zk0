"""Unit tests for server_app.py functions."""

import pytest
from unittest.mock import Mock, patch
from src.server_app import (
    evaluate_model_on_datasets,
    evaluate_single_dataset,
    should_skip_evaluation,
    prepare_evaluation_model,
    process_evaluation_metrics,
    log_evaluation_to_wandb,
    save_evaluation_results,
    generate_evaluation_charts,
)
from src.server.server_utils import (
    aggregate_client_metrics,
    collect_individual_client_metrics,
    aggregate_and_log_metrics,
    save_and_push_model,
    finalize_round_metrics,
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




class TestAggregateParameters:
    """Test the aggregate_parameters function."""

    def test_aggregate_parameters_success(self):
        """Test successful parameter aggregation."""
        from unittest.mock import Mock, patch
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
        from unittest.mock import Mock, patch
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
        with patch("src.server.server_utils.compute_server_param_update_norm") as mock_compute_norm:
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
        from unittest.mock import Mock

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


class TestSaveAndPushModel:
    """Test the save_and_push_model function."""

    def test_save_and_push_model_checkpoint_only(self):
        """Test saving checkpoint without pushing to hub."""
        from unittest.mock import Mock, patch
        from flwr.common import ndarrays_to_parameters
        import numpy as np

        # Create mock strategy
        strategy = Mock()
        strategy.context.run_config = {"checkpoint_interval": 5}
        strategy.models_dir = Mock()
        strategy.num_rounds = 5  # Make it final round to trigger hub logic

        server_round = 5  # Final round
        aggregated_parameters = ndarrays_to_parameters([np.array([1.0, 2.0])])

        with patch("src.server.server_utils.save_model_checkpoint") as mock_save, \
             patch("src.server.server_utils.logger") as mock_logger:
            save_and_push_model(strategy, server_round, aggregated_parameters)

            # Verify checkpoint was saved once (final round)
            mock_save.assert_called_once_with(strategy, aggregated_parameters, server_round)

            # Verify no hub push (no hf_repo_id in config) - skip message logged in final block
            mock_logger.info.assert_any_call(
                "ℹ️ Server: No hf_repo_id configured, skipping Hub push"
            )

    def test_save_and_push_model_with_hub_push(self):
        """Test saving checkpoint and pushing to hub."""
        from unittest.mock import Mock, patch
        from flwr.common import ndarrays_to_parameters
        import numpy as np

        # Create mock strategy
        strategy = Mock()
        strategy.context.run_config = {
            "checkpoint_interval": 5,
            "hf_repo_id": "test/repo"
        }
        strategy.models_dir = Mock()
        strategy.num_rounds = 10

        server_round = 10  # Final round
        aggregated_parameters = ndarrays_to_parameters([np.array([1.0, 2.0])])

        with patch("src.server.server_utils.save_model_checkpoint") as mock_save, \
             patch("src.server.server_utils.push_model_to_hub_enhanced") as mock_push, \
             patch("src.server.server_utils.logger") as mock_logger:
            mock_save.return_value = "/path/to/checkpoint"  # Mock return value
            save_and_push_model(strategy, server_round, aggregated_parameters)

            # Verify checkpoint was saved once (final round, deduplicated)
            mock_save.assert_called_once_with(strategy, aggregated_parameters, server_round)

            # Verify hub push was attempted
            mock_push.assert_called_once_with("/path/to/checkpoint", "test/repo")

    def test_save_and_push_model_skip_hub_push_when_rounds_less_than_interval(self):
        """Test skipping hub push when num_rounds < checkpoint_interval."""
        from unittest.mock import Mock, patch
        from flwr.common import ndarrays_to_parameters
        import numpy as np

        # Create mock strategy
        strategy = Mock()
        strategy.context.run_config = {
            "checkpoint_interval": 20,
            "hf_repo_id": "test/repo"
        }
        strategy.models_dir = Mock()
        strategy.num_rounds = 5  # Less than checkpoint_interval

        server_round = 5  # Final round
        aggregated_parameters = ndarrays_to_parameters([np.array([1.0, 2.0])])

        with patch("src.server.server_utils.save_model_checkpoint") as mock_save, \
             patch("src.server.server_utils.push_model_to_hub_enhanced") as mock_push, \
             patch("src.server.server_utils.logger") as mock_logger:
            save_and_push_model(strategy, server_round, aggregated_parameters)

            # Verify checkpoint was saved (always saved on final round)
            mock_save.assert_called_once_with(strategy, aggregated_parameters, server_round)

            # Verify hub push was skipped due to num_rounds < checkpoint_interval
            mock_push.assert_not_called()

            # Verify the skip message was logged
            mock_logger.info.assert_any_call(
                "ℹ️ Server: Skipping HF Hub push - num_rounds (5) < checkpoint_interval (20)"
            )


class TestComputeFedproxParameters:
    """Test the compute_fedprox_parameters method."""

    def test_fixed_adjustment_disabled(self):
        """Test fixed adjustment when dynamic_training_decay is disabled."""
        from unittest.mock import Mock
        from src.server_app import AggregateEvaluationStrategy

        # Create mock strategy
        strategy = Mock(spec=AggregateEvaluationStrategy)
        strategy.proximal_mu = 0.01
        strategy.current_lr = 0.0005  # Simulate existing LR

        # Create test instance
        test_strategy = AggregateEvaluationStrategy.__new__(AggregateEvaluationStrategy)
        test_strategy.proximal_mu = 0.01
        test_strategy.current_lr = 0.0005

        # Test with dynamic_training_decay disabled
        app_config = {"dynamic_training_decay": False, "initial_lr": 0.0005}

        current_mu, current_lr = test_strategy.compute_fedprox_parameters(5, app_config)

        # Should use fixed halving: mu / (2^(round//10)) = 0.01 / (2^(5//10)) = 0.01 / 2^0 = 0.01
        assert current_mu == 0.01
        assert current_lr == 0.0005  # LR unchanged

    def test_fixed_adjustment_round_10(self):
        """Test fixed adjustment at round 10 (first halving)."""
        from unittest.mock import Mock
        from src.server_app import AggregateEvaluationStrategy

        # Create test instance
        test_strategy = AggregateEvaluationStrategy.__new__(AggregateEvaluationStrategy)
        test_strategy.proximal_mu = 0.01
        test_strategy.current_lr = 0.0005

        # Test at round 10
        app_config = {"dynamic_training_decay": False, "initial_lr": 0.0005}

        current_mu, current_lr = test_strategy.compute_fedprox_parameters(
            10, app_config
        )

        # Should use fixed halving: mu / (2^(10//10)) = 0.01 / (2^1) = 0.005
        assert current_mu == 0.005
        assert current_lr == 0.0005  # LR unchanged

    def test_dynamic_adjustment_stall(self):
        """Test dynamic adjustment for stall scenario."""
        from unittest.mock import Mock, patch
        from src.server_app import AggregateEvaluationStrategy

        # Create test instance
        test_strategy = AggregateEvaluationStrategy.__new__(AggregateEvaluationStrategy)
        test_strategy.proximal_mu = 0.01
        test_strategy.current_lr = 0.0005
        test_strategy.server_eval_losses = [1.0, 0.995, 0.994]  # Stall pattern

        # Test with dynamic_training_decay enabled
        app_config = {"dynamic_training_decay": True, "initial_lr": 0.0005}

        current_mu, current_lr = test_strategy.compute_fedprox_parameters(5, app_config)

        # Should use joint adjustment for stall: both mu and lr reduced by 0.8
        assert current_mu == 0.008  # 0.01 * 0.8
        assert current_lr == 0.0004  # 0.0005 * 0.8
        assert test_strategy.current_lr == 0.0004  # Should be updated for next round

    def test_dynamic_adjustment_insufficient_data(self):
        """Test dynamic adjustment with insufficient evaluation data."""
        from unittest.mock import Mock
        from src.server_app import AggregateEvaluationStrategy

        # Create test instance
        test_strategy = AggregateEvaluationStrategy.__new__(AggregateEvaluationStrategy)
        test_strategy.proximal_mu = 0.01
        test_strategy.current_lr = 0.0005
        test_strategy.server_eval_losses = [1.0, 0.95]  # Only 2 losses

        # Test with dynamic_training_decay enabled but insufficient data
        app_config = {"dynamic_training_decay": True, "initial_lr": 0.0005}

        current_mu, current_lr = test_strategy.compute_fedprox_parameters(5, app_config)

        # Should fall back to fixed adjustment
        assert current_mu == 0.01  # No change
        assert current_lr == 0.0005  # No change

    def test_lr_initialization(self):
        """Test that current_lr is initialized from config when not set."""
        from unittest.mock import Mock
        from src.server_app import AggregateEvaluationStrategy

        # Create test instance without current_lr set
        test_strategy = AggregateEvaluationStrategy.__new__(AggregateEvaluationStrategy)
        test_strategy.proximal_mu = 0.01
        # Don't set current_lr - should be initialized from config

        app_config = {"dynamic_training_decay": False, "initial_lr": 0.001}

        current_mu, current_lr = test_strategy.compute_fedprox_parameters(1, app_config)

        assert current_mu == 0.01
        assert current_lr == 0.001  # Should be initialized from config
        assert test_strategy.current_lr == 0.001  # Should be set for future rounds


class TestEvaluateModelOnDatasets:
    """Test the evaluate_model_on_datasets function."""

    def test_evaluate_model_on_datasets_empty_config(self):
        """Test evaluation with empty dataset config."""
        from unittest.mock import Mock, patch
        import numpy as np

        # Mock model and device
        model = Mock()
        device = Mock()

        # Empty config
        datasets_config = []

        # Mock test function to return fixed values
        with patch("src.task.test") as mock_test:
            mock_test.return_value = (1.0, 100, {"policy_loss": 1.0})

            composite_loss, total_examples, composite_metrics, per_dataset_results = (
                evaluate_model_on_datasets(
                    model, datasets_config, device, eval_batches=8
                )
            )

            # Should return zeros for empty config
            assert composite_loss == 0.0
            assert total_examples == 0
            assert composite_metrics["composite_eval_loss"] == 0.0
            assert composite_metrics["num_datasets_evaluated"] == 0
            assert per_dataset_results == []

    def test_evaluate_single_dataset(self):
        """Test evaluation of a single dataset."""
        import numpy as np
        from unittest.mock import Mock

        # Mock dependencies
        global_parameters = [np.array([1.0, 2.0])]
        dataset_name = "test_dataset"
        evaldata_id = 123
        device = Mock()
        eval_batches = 8

        # Mock functions
        mock_dataset = Mock()
        mock_dataset.meta = Mock()
        mock_policy = Mock()
        mock_metrics = {"policy_loss": 0.5, "action_dim": 7}

        load_lerobot_dataset_fn = Mock(return_value=mock_dataset)
        make_policy_fn = Mock(return_value=mock_policy)
        set_params_fn = Mock()
        test_fn = Mock(return_value=(0.5, 100, mock_metrics))

        result = evaluate_single_dataset(
            global_parameters=global_parameters,
            dataset_name=dataset_name,
            evaldata_id=evaldata_id,
            device=device,
            eval_batches=eval_batches,
            load_lerobot_dataset_fn=load_lerobot_dataset_fn,
            make_policy_fn=make_policy_fn,
            set_params_fn=set_params_fn,
            test_fn=test_fn,
        )

        # Verify function calls
        load_lerobot_dataset_fn.assert_called_once_with(dataset_name)
        make_policy_fn.assert_called_once()
        set_params_fn.assert_called_once_with(mock_policy, global_parameters)
        test_fn.assert_called_once_with(
            mock_policy, device=device, eval_batches=eval_batches, dataset=mock_dataset
        )

        # Verify result structure
        assert result["dataset_name"] == dataset_name
        assert result["evaldata_id"] == evaldata_id
        assert result["loss"] == 0.5
        assert result["num_examples"] == 100
        assert result["metrics"] == mock_metrics

    def test_evaluate_model_on_datasets_single_dataset(self):
        """Test evaluation with single dataset."""
        import numpy as np
        import torch
        from unittest.mock import Mock, patch

        # Mock ServerConfig
        class MockServerConfig:
            def __init__(self, name, evaldata_id=None):
                self.name = name
                self.evaldata_id = evaldata_id

        # Mock dependencies
        global_parameters = [np.array([1.0, 2.0, 3.0])]  # Match state_dict size
        datasets_config = [MockServerConfig("test_dataset", 123)]
        device = torch.device("cpu")
        eval_batches = 8

        # Mock functions and objects
        mock_dataset = Mock()
        mock_dataset.meta = {"observation_space": Mock(), "action_space": Mock()}  # Proper meta dict
        mock_policy = Mock()
        mock_policy.state_dict.return_value = {'policy.weight': torch.zeros(3, dtype=torch.float32)}  # For set_params
        mock_metrics = {"policy_loss": 0.5, "action_dim": 7}

        load_lerobot_dataset_fn = Mock(return_value=mock_dataset)
        make_policy_fn = Mock(return_value=mock_policy)
        set_params_fn = Mock()
        test_fn = Mock(return_value=(0.5, 100, mock_metrics))

        # Patch the imports in evaluate_model_on_datasets - patch server_app.set_params for the passed value
        with (
            patch("src.task.test", test_fn),
            patch("src.server_app.set_params", set_params_fn),
            patch("src.utils.load_lerobot_dataset", load_lerobot_dataset_fn),
            patch("lerobot.policies.factory.make_policy", make_policy_fn),
        ):
            composite_loss, total_examples, composite_metrics, per_dataset_results = (
                evaluate_model_on_datasets(
                    global_parameters, datasets_config, device, eval_batches
                )
            )

        # Verify results
        assert composite_loss == 0.5
        assert total_examples == 100
        assert composite_metrics["composite_eval_loss"] == 0.5
        assert composite_metrics["num_datasets_evaluated"] == 1
        assert len(per_dataset_results) == 1

        result = per_dataset_results[0]
        assert result["dataset_name"] == "test_dataset"
        assert result["evaldata_id"] == 123
        assert result["loss"] == 0.5
        assert result["num_examples"] == 100
        assert result["metrics"] == mock_metrics

    def test_evaluate_model_on_datasets_multiple_datasets(self):
        """Test evaluation with multiple datasets."""
        import numpy as np
        import torch
        from unittest.mock import Mock, patch

        # Mock ServerConfig
        class MockServerConfig:
            def __init__(self, name, evaldata_id=None):
                self.name = name
                self.evaldata_id = evaldata_id

        # Mock dependencies
        global_parameters = [np.array([1.0, 2.0, 3.0])]  # Match state_dict size
        datasets_config = [
            MockServerConfig("dataset1", 123),
            MockServerConfig("dataset2", 456),
        ]
        device = torch.device("cpu")
        eval_batches = 8

        # Mock functions and objects
        mock_dataset1 = Mock()
        mock_dataset1.meta = {"observation_space": Mock(), "action_space": Mock()}  # Proper meta dict
        mock_dataset2 = Mock()
        mock_dataset2.meta = {"observation_space": Mock(), "action_space": Mock()}  # Proper meta dict
        mock_policy1 = Mock()
        mock_policy1.state_dict.return_value = {'policy.weight': torch.zeros(3, dtype=torch.float32)}  # For set_params
        mock_policy2 = Mock()
        mock_policy2.state_dict.return_value = {'policy.weight': torch.zeros(3, dtype=torch.float32)}  # For set_params
        mock_metrics1 = {"policy_loss": 0.5, "action_dim": 7}
        mock_metrics2 = {"policy_loss": 1.0, "action_dim": 7}

        load_lerobot_dataset_fn = Mock(side_effect=[mock_dataset1, mock_dataset2])
        make_policy_fn = Mock(side_effect=[mock_policy1, mock_policy2])
        set_params_fn = Mock()
        test_fn = Mock(
            side_effect=[(0.5, 100, mock_metrics1), (1.0, 150, mock_metrics2)]
        )

        # Patch the imports in evaluate_model_on_datasets - patch server_app.set_params for the passed value
        with (
            patch("src.task.test", test_fn),
            patch("src.server_app.set_params", set_params_fn),
            patch("src.utils.load_lerobot_dataset", load_lerobot_dataset_fn),
            patch("lerobot.policies.factory.make_policy", make_policy_fn),
        ):
            composite_loss, total_examples, composite_metrics, per_dataset_results = (
                evaluate_model_on_datasets(
                    global_parameters, datasets_config, device, eval_batches
                )
            )

        # Verify results
        assert composite_loss == 0.75  # Average of 0.5 and 1.0
        assert total_examples == 250  # Sum of 100 and 150
        assert composite_metrics["composite_eval_loss"] == 0.75
        assert composite_metrics["num_datasets_evaluated"] == 2
        assert len(per_dataset_results) == 2

        # Check first dataset result
        result1 = per_dataset_results[0]
        assert result1["dataset_name"] == "dataset1"
        assert result1["evaldata_id"] == 123
        assert result1["loss"] == 0.5
        assert result1["num_examples"] == 100

        # Check second dataset result
        result2 = per_dataset_results[1]
        assert result2["dataset_name"] == "dataset2"
        assert result2["evaldata_id"] == 456
        assert result2["loss"] == 1.0
        assert result2["num_examples"] == 150

        # Check composite metrics include per-dataset losses
        assert composite_metrics["loss_evaldata_id_123"] == 0.5
        assert composite_metrics["loss_evaldata_id_456"] == 1.0


class TestShouldSkipEvaluation:
    """Test the should_skip_evaluation function."""

    def test_should_skip_when_multiple_of_frequency(self):
        """Test that evaluation is not skipped when round is multiple of frequency."""
        result = should_skip_evaluation(server_round=5, eval_frequency=5)
        assert result is False

    def test_should_skip_when_not_multiple_of_frequency(self):
        """Test that evaluation is skipped when round is not multiple of frequency."""
        result = should_skip_evaluation(server_round=6, eval_frequency=5)
        assert result is True

    def test_should_skip_frequency_one(self):
        """Test with frequency of 1 (evaluate every round)."""
        result = should_skip_evaluation(server_round=3, eval_frequency=1)
        assert result is False

    def test_should_skip_round_zero(self):
        """Test with round 0."""
        result = should_skip_evaluation(server_round=0, eval_frequency=2)
        assert result is False  # 0 % 2 == 0, so should not skip


class TestPrepareEvaluationModel:
    """Test the prepare_evaluation_model function."""

    def test_prepare_evaluation_model(self):
        """Test model preparation for evaluation."""
        import torch
        import numpy as np
        from unittest.mock import Mock, patch

        # Mock parameters and device as proper NDArrays
        parameters = [np.array([1.0, 2.0, 3.0])]  # Proper numpy array list
        device = torch.device("cpu")
        template_model = Mock()
        # Mock parameters() to return proper tensors for numel
        mock_param = Mock()
        mock_param.numel.return_value = 100
        template_model.parameters.return_value = [mock_param]
        template_model.to.return_value = template_model  # Return self for chaining
        template_model.state_dict.return_value = {'policy.weight': torch.zeros(3, dtype=torch.float32)}  # For set_params

        # Mock set_params in server_app namespace (since prepare calls it directly)
        with patch("src.server_app.set_params") as mock_set_params:
            result = prepare_evaluation_model(parameters, device, template_model)

            # Verify set_params was called with proper parameters
            mock_set_params.assert_called_once_with(template_model, parameters)

            # Verify model.to was called
            template_model.to.assert_called_once_with(device)

            # Verify the model returned by to() is returned
            assert result == template_model

            # Verify numel was called (if used in logging)
            mock_param.numel.assert_called_once()


class TestProcessEvaluationMetrics:
    """Test the process_evaluation_metrics function."""

    def test_process_evaluation_metrics(self):
        """Test processing of evaluation metrics."""
        from unittest.mock import Mock

        # Mock strategy
        strategy = Mock()
        strategy.federated_metrics_history = []
        strategy.server_eval_losses = []  # Initialize as list

        server_round = 5
        loss = 0.8
        metrics = {"policy_loss": 0.8}
        aggregated_client_metrics = {"num_clients": 3, "avg_client_loss": 1.2}
        individual_client_metrics = [{"client_id": "client_0"}]

        process_evaluation_metrics(
            strategy,
            server_round,
            loss,
            metrics,
            aggregated_client_metrics,
            individual_client_metrics,
        )

        # Verify metrics were added to history
        assert len(strategy.federated_metrics_history) == 1
        round_metrics = strategy.federated_metrics_history[0]
        assert round_metrics["round"] == server_round
        assert round_metrics["num_clients"] == 3
        assert round_metrics["avg_policy_loss"] == 0.8
        assert round_metrics["avg_client_loss"] == 1.2
        assert "round_time" not in round_metrics  # Should not have round_time field


class TestLogEvaluationToWandb:
    """Test the log_evaluation_to_wandb function."""

    def test_log_evaluation_to_wandb_with_run(self):
        """Test WandB logging when run exists."""
        from unittest.mock import Mock, patch

        # Mock strategy with wandb run
        strategy = Mock()
        strategy.wandb_run = Mock()

        server_round = 5
        loss = 0.8
        metrics = {"policy_loss": 0.8}
        aggregated_client_metrics = {"num_clients": 3}
        individual_client_metrics = [{"client_id": "client_0"}]

        with patch("src.wandb_utils.log_wandb_metrics") as mock_log:
            with patch("src.utils.prepare_server_wandb_metrics") as mock_prepare:
                mock_prepare.return_value = {"test_metric": 1.0}

                log_evaluation_to_wandb(
                    strategy,
                    server_round,
                    loss,
                    metrics,
                    aggregated_client_metrics,
                    individual_client_metrics,
                )

                # Verify prepare and log were called
                mock_prepare.assert_called_once()
                mock_log.assert_called_once_with(
                    {"test_metric": 1.0}, step=server_round
                )

    def test_log_evaluation_to_wandb_no_run(self):
        """Test WandB logging when no run exists."""
        from unittest.mock import Mock, patch

        # Mock strategy without wandb run
        strategy = Mock()
        strategy.wandb_run = None

        server_round = 5
        loss = 0.8
        metrics = {"policy_loss": 0.8}
        aggregated_client_metrics = {"num_clients": 3}
        individual_client_metrics = [{"client_id": "client_0"}]

        with patch("src.wandb_utils.log_wandb_metrics") as mock_log:
            log_evaluation_to_wandb(
                strategy,
                server_round,
                loss,
                metrics,
                aggregated_client_metrics,
                individual_client_metrics,
            )

            # Verify log was not called
            mock_log.assert_not_called()

    def test_log_evaluation_to_wandb_with_per_dataset_results(self):
        """Test WandB logging with per-dataset results passed through."""
        from unittest.mock import Mock, patch

        # Mock strategy with wandb run
        strategy = Mock()
        strategy.wandb_run = Mock()

        server_round = 49
        loss = 0.22341731128593287
        metrics = {"policy_loss": 0.20069279242306948}
        aggregated_client_metrics = {"num_clients": 3}
        individual_client_metrics = [{"client_id": "client_3"}]

        # Mock per_dataset_results matching JSON structure
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

        with patch("src.wandb_utils.log_wandb_metrics") as mock_log:
            with patch("src.utils.prepare_server_wandb_metrics") as mock_prepare:
                mock_prepare.return_value = {"loss_evaldata_id_0": 0.20069279242306948}

                log_evaluation_to_wandb(
                    strategy,
                    server_round,
                    loss,
                    metrics,
                    aggregated_client_metrics,
                    individual_client_metrics,
                    per_dataset_results,
                )

                # Verify prepare was called with per_dataset_results
                mock_prepare.assert_called_once_with(
                    server_round=server_round,
                    server_loss=loss,
                    server_metrics=metrics,
                    aggregated_client_metrics=aggregated_client_metrics,
                    individual_client_metrics=individual_client_metrics,
                    per_dataset_results=per_dataset_results,
                )
                mock_log.assert_called_once_with(
                    {"loss_evaldata_id_0": 0.20069279242306948}, step=server_round
                )


class TestSaveEvaluationResults:
    """Test the save_evaluation_results function."""

    def test_save_evaluation_results(self):
        """Test saving evaluation results to file."""
        from unittest.mock import Mock, patch
        from pathlib import Path
        import json
        import tempfile
        import os

        # Use a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock strategy
            strategy = Mock()
            strategy.server_dir = Path(temp_dir)
            strategy.server_eval_losses = [0.8]  # Set to list to avoid Mock subscript error
    
            server_round = 5
            loss = 0.8
            num_examples = 1000
            metrics = {"policy_loss": 0.8}
            aggregated_client_metrics = {"num_clients": 3}
            individual_client_metrics = [{"client_id": "client_0", "round": 0}]
    
            # Set strategy attributes after variable definitions
            strategy.last_aggregated_metrics = aggregated_client_metrics
            strategy.last_client_metrics = individual_client_metrics
            strategy.last_per_dataset_results = []

            save_evaluation_results(
                strategy,
                server_round,
                loss,
                num_examples,
                metrics,
                aggregated_client_metrics,
                individual_client_metrics,
            )

            # Verify file was created and contains correct data
            expected_file = Path(temp_dir) / f"round_{server_round}_server_eval.json"
            assert expected_file.exists()

            with open(expected_file, "r") as f:
                written_data = json.load(f)

            # Verify structure matches current prepare_server_eval_metrics implementation
            assert written_data["composite_eval_loss"] == loss
            assert written_data["num_datasets_evaluated"] == 0  # No per_dataset_results
            assert written_data["aggregated_client_metrics"] == aggregated_client_metrics
            assert written_data["individual_client_metrics"] == individual_client_metrics
            assert written_data["server_eval_dataset_results"] == []


class TestGenerateEvaluationCharts:
    """Test the generate_evaluation_charts function."""

    def test_generate_evaluation_charts_final_round(self):
        """Test chart generation on final round."""
        from unittest.mock import Mock, patch
        from pathlib import Path

        # Mock strategy
        strategy = Mock()
        strategy.num_rounds = 10
        strategy.server_dir = Path("/tmp")  # Use real Path object
        strategy.wandb_run = Mock()

        server_round = 10

        with patch("src.visualization.SmolVLAVisualizer") as mock_visualizer_class:
            with patch(
                "src.server_app.aggregate_eval_policy_loss_history"
            ) as mock_aggregate:
                mock_aggregate.return_value = {"0": {"server_policy_loss": 1.0}}
                mock_visualizer = Mock()
                mock_visualizer_class.return_value = mock_visualizer

                generate_evaluation_charts(strategy, server_round)

                # Verify visualizer was created and methods called
                mock_visualizer_class.assert_called_once()
                mock_visualizer.plot_eval_policy_loss_chart.assert_called_once()
                mock_visualizer.plot_federated_metrics.assert_called_once()

    def test_generate_evaluation_charts_not_final_round(self):
        """Test no chart generation when not final round."""
        from unittest.mock import Mock, patch

        # Mock strategy
        strategy = Mock()
        strategy.num_rounds = 10
        strategy.server_dir = Mock()

        server_round = 5

        with patch("src.visualization.SmolVLAVisualizer") as mock_visualizer_class:
            generate_evaluation_charts(strategy, server_round)

            # Verify visualizer was not created
            mock_visualizer_class.assert_not_called()




class TestPushModelToHub:
    """Test the push_model_to_hub method."""

    def test_push_model_to_hub_no_token(self):
        """Test push failure when HF_TOKEN is missing."""
        from unittest.mock import Mock, patch
        import os
        import torch
        from src.server_app import AggregateEvaluationStrategy
        import numpy as np
        from flwr.common import ndarrays_to_parameters

        # Create a minimal strategy instance for testing
        strategy = AggregateEvaluationStrategy.__new__(AggregateEvaluationStrategy)
        strategy.models_dir = Mock()
        strategy.template_model = Mock()  # Mock template model

        # Create mock param with proper attributes to avoid conversion errors
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.shape = (2,)
        strategy.template_model.state_dict.return_value = {"param1": mock_param}

        # Create proper Flower parameters
        ndarrays = [np.array([1.0, 2.0])]
        parameters = ndarrays_to_parameters(ndarrays)

        server_round = 250
        hf_repo_id = "test/repo"

        # Remove HF_TOKEN from environment and patch the parameter conversion to avoid errors
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.server_app.logger") as mock_logger:
                with patch("huggingface_hub.HfApi") as mock_hf_api:
                    # Make the mock raise the expected ValueError when HF_TOKEN is missing
                    mock_hf_api.side_effect = ValueError(
                        "HF_TOKEN environment variable not found"
                    )
                    # Should raise ValueError
                    with pytest.raises(
                        ValueError, match="HF_TOKEN environment variable not found"
                    ):
                        strategy.push_model_to_hub(parameters, server_round, hf_repo_id)
