"""Unit tests for evaluation functions."""

from unittest.mock import Mock, patch
from src.server.evaluation import (
    evaluate_model_on_datasets,
    evaluate_single_dataset,
    should_skip_evaluation,
    prepare_evaluation_model,
    process_evaluation_metrics,
    log_evaluation_to_wandb,
    save_evaluation_results,
)


class TestEvaluateModelOnDatasets:
    """Test the evaluate_model_on_datasets function."""

    def test_evaluate_model_on_datasets_empty_config(self):
        """Test evaluation with empty dataset config."""

        # Mock model and device
        model = Mock()
        device = Mock()

        # Empty config
        datasets_config = []

        # Mock test function to return fixed values
        with patch("src.training.evaluation.test") as mock_test:
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

        # Patch the imports in evaluate_model_on_datasets
        with (
            patch("src.server.evaluation.test", test_fn),
            patch("src.server.evaluation.set_params", set_params_fn),
            patch("src.server.evaluation.load_lerobot_dataset", load_lerobot_dataset_fn),
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

        # Patch the imports in evaluate_model_on_datasets
        with (
            patch("src.server.evaluation.test", test_fn),
            patch("src.server.evaluation.set_params", set_params_fn),
            patch("src.server.evaluation.load_lerobot_dataset", load_lerobot_dataset_fn),
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

        # Mock parameters and device as proper NDArrays
        parameters = [np.array([1.0, 2.0, 3.0])]  # Proper numpy array list
        device = torch.device("cpu")
        template_model = Mock()
        # Mock parameters() to return proper tensors for numel
        mock_param = Mock()
        mock_param.numel.return_value = 100
        template_model.parameters.return_value = [mock_param]
        template_model.to.return_value = template_model  # Return self for chaining
        # Mock state_dict to return proper tensors
        mock_tensor = Mock()
        mock_tensor.norm.return_value = Mock()  # Mock the norm method
        template_model.state_dict.return_value = {'policy.weight': mock_tensor}

        # Mock set_params in evaluation namespace
        with patch("src.server.evaluation.set_params") as mock_set_params:
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

        # Mock strategy with wandb run
        strategy = Mock()
        strategy.wandb_run = Mock()  # This should make it truthy

        server_round = 5
        loss = 0.8
        metrics = {"policy_loss": 0.8}
        aggregated_client_metrics = {"num_clients": 3}
        individual_client_metrics = [{"client_id": "client_0"}]

        with patch("src.server.evaluation.log_wandb_metrics") as mock_log:
            with patch("src.server.evaluation.prepare_server_wandb_metrics") as mock_prepare:
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

        # Mock strategy without wandb run
        strategy = Mock()
        strategy.wandb_run = None

        server_round = 5
        loss = 0.8
        metrics = {"policy_loss": 0.8}
        aggregated_client_metrics = {"num_clients": 3}
        individual_client_metrics = [{"client_id": "client_0"}]

        with patch("src.server.wandb_utils.log_wandb_metrics") as mock_log:
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

        with patch("src.server.evaluation.log_wandb_metrics") as mock_log:
            with patch("src.server.evaluation.prepare_server_wandb_metrics") as mock_prepare:
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

                # Verify prepare was called with per_eval_dataset_results
                mock_prepare.assert_called_once_with(
                    server_round=server_round,
                    server_loss=loss,
                    server_metrics=metrics,
                    aggregated_client_metrics=aggregated_client_metrics,
                    individual_client_metrics=individual_client_metrics,
                    per_eval_dataset_results=per_dataset_results,
                )
                mock_log.assert_called_once_with(
                    {"loss_evaldata_id_0": 0.20069279242306948}, step=server_round
                )


class TestSaveEvaluationResults:
    """Test the save_evaluation_results function."""

    def test_save_evaluation_results(self):
        """Test saving evaluation results to file."""
        from pathlib import Path
        import json
        import tempfile

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
            assert written_data["server_composite_eval_loss"] == loss
            assert written_data["num_server_eval_datasets"] == 0  # No per_dataset_results
            assert written_data["client_aggregated_training_metrics"] == aggregated_client_metrics
            assert written_data["individual_client_training_metrics"] == individual_client_metrics
            assert written_data["server_per_dataset_eval_results"] == []

