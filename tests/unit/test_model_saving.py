"""Unit tests for model saving and pushing functions."""

import pytest
from unittest.mock import Mock, patch
from src.server.model_checkpointing import save_and_push_model


class TestSaveAndPushModel:
    """Test the save_and_push_model function."""

    def test_save_and_push_model_checkpoint_only(self):
        """Test saving checkpoint without pushing to hub."""
        from flwr.common import ndarrays_to_parameters
        import numpy as np

        # Create mock strategy
        from pathlib import Path
        strategy = Mock()
        strategy.context.run_config = {"checkpoint_interval": 5}
        strategy.models_dir = Path("/tmp/mock")
        strategy.num_rounds = 5  # Make it final round to trigger hub logic
        strategy.template_model = Mock()
        strategy.template_model.state_dict.return_value = {"param1": Mock(), "param2": Mock()}
        strategy.template_model.config.save_pretrained = Mock()

        server_round = 5  # Final round
        aggregated_parameters = ndarrays_to_parameters([np.array([1.0, 2.0])])

        with patch("src.server.model_utils.save_model_checkpoint") as mock_save, \
              patch("src.server.model_checkpointing.logger") as mock_logger, \
              patch("src.core.utils.get_tool_config") as mock_get_config:
            mock_get_config.return_value = {"app": {"config": {"checkpoint_interval": 5}}}
            save_and_push_model(strategy, server_round, aggregated_parameters, {})

            # Verify checkpoint was saved once (final round)
            mock_save.assert_called_once_with(strategy, aggregated_parameters, server_round)

            # Verify no hub push (no hf_repo_id in config) - skip message logged in final block
            mock_logger.info.assert_any_call(
                "ℹ️ Server: No hf_repo_id configured, skipping Hub push"
            )

    def test_save_and_push_model_with_hub_push(self):
        """Test saving checkpoint and pushing to hub."""
        from flwr.common import ndarrays_to_parameters
        import numpy as np

        # Create mock strategy
        from pathlib import Path
        strategy = Mock()
        strategy.context.run_config = {
            "checkpoint_interval": 5,
            "hf_repo_id": "test/repo"
        }
        strategy.models_dir = Path("/tmp/mock")
        strategy.num_rounds = 10

        server_round = 10  # Final round
        aggregated_parameters = ndarrays_to_parameters([np.array([1.0, 2.0])])

        with patch("src.server.model_utils.save_model_checkpoint") as mock_save, \
                  patch("src.server.model_utils.push_model_to_hub_enhanced") as mock_push, \
                  patch("src.core.utils.get_tool_config") as mock_get_config:
            from pathlib import Path
            mock_get_config.return_value = {"app": {"config": {"checkpoint_interval": 5, "hf_repo_id": "test/repo"}}}
            mock_save.return_value = Path("/path/to/checkpoint")  # Mock return value
            save_and_push_model(strategy, server_round, aggregated_parameters, {})

            # Verify checkpoint was saved once (final round, deduplicated)
            mock_save.assert_called_once_with(strategy, aggregated_parameters, server_round)

            # Verify hub push was attempted
            from pathlib import Path
            mock_push.assert_called_once_with(Path("/path/to/checkpoint"), "test/repo")

    def test_save_and_push_model_skip_hub_push_when_rounds_less_than_interval(self):
        """Test skipping hub push when num_rounds < checkpoint_interval."""
        from flwr.common import ndarrays_to_parameters
        import numpy as np

        # Create mock strategy
        from pathlib import Path
        strategy = Mock()
        strategy.context.run_config = {
            "checkpoint_interval": 20,
            "hf_repo_id": "test/repo"
        }
        strategy.models_dir = Path("/tmp/mock")
        strategy.num_rounds = 5  # Less than checkpoint_interval

        server_round = 5  # Final round
        aggregated_parameters = ndarrays_to_parameters([np.array([1.0, 2.0])])

        with patch("src.server.model_utils.save_model_checkpoint") as mock_save, \
                  patch("src.server.model_utils.push_model_to_hub_enhanced") as mock_push:
            save_and_push_model(strategy, server_round, aggregated_parameters, {})

            # Verify checkpoint was saved (always saved on final round)
            mock_save.assert_called_once_with(strategy, aggregated_parameters, server_round)

            # Verify hub push was skipped due to num_rounds < checkpoint_interval
            mock_push.assert_not_called()


class TestPushModelToHub:
    """Test the push_model_to_hub method."""

    def test_push_model_to_hub_no_token(self):
        """Test push failure when HF_TOKEN is missing."""
        import os
        import torch
        from src.server.strategy import AggregateEvaluationStrategy
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
            with patch("src.server_app.logger"):
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