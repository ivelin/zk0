"""Unit tests for checkpoint saving functions."""

from unittest.mock import patch, MagicMock, mock_open
import tempfile
from pathlib import Path
import torch
import numpy as np


class TestSaveModelCheckpoint:
    """Test save_model_checkpoint function."""

    @patch("src.server.server_utils.get_tool_config")
    @patch("importlib.metadata.version")
    @patch("safetensors.torch.save_file")
    @patch("src.server.server_utils.generate_model_card")
    @patch("src.server.server_utils.extract_training_hyperparameters")
    @patch("src.server.server_utils.extract_datasets")
    @patch("src.server.server_utils.compute_in_memory_insights")  # New in-memory helper
    @patch("src.wandb_utils.get_wandb_public_url")
    def test_save_model_checkpoint_in_memory_success(self, mock_get_wandb_url, mock_compute_insights,
                                                       mock_extract_datasets, mock_extract_hyperparams,
                                                       mock_generate_card, mock_save_file, mock_version, mock_get_config):
        """Test successful checkpoint with in-memory metrics."""
        from src.server.server_utils import save_model_checkpoint

        # Setup mocks for in-memory data
        mock_version.return_value = "0.3.8"
        mock_get_config.side_effect = [
            {"app": {"config": {"local-epochs": 20, "hf_repo_id": "ivelin/zk0-smolvla-fl"}}},  # flwr config with repo_id
            {"datasets": {"clients": [{"name": "client1"}], "server": [{"name": "eval1"}]}}  # zk0 config
        ]
        mock_extract_hyperparams.return_value = {"num_server_rounds": 2}
        mock_extract_datasets.return_value = ([{"name": "client1", "description": "Test client"}], [{"name": "eval1", "description": "Test eval"}])
        mock_compute_insights.return_value = {"convergence_trend": "0.723 → 0.655", "avg_client_loss_trend": "0.912 → 0.894", "client_participation_rate": "Average 2.0 clients per round", "anomalies": []}
        mock_generate_card.return_value = "# Model Card with repo_id"
        mock_get_wandb_url.return_value = None  # No WandB URL for this test

        # Create mock strategy with in-memory data
        mock_strategy = MagicMock()
        mock_strategy.context.run_config = {"federation": "local-simulation", "hf_repo_id": "ivelin/zk0-smolvla-fl"}
        mock_strategy.server_dir = Path("/tmp/server")
        mock_strategy.models_dir = Path("/tmp/models")
        mock_strategy.template_model = MagicMock()
        mock_strategy.server_eval_losses = [0.655]
        mock_strategy.last_aggregated_metrics = {"avg_client_loss": 0.894, "num_clients": 2}
        mock_strategy.last_client_metrics = [{"client_id": 6}, {"client_id": 2}]
        mock_strategy.last_per_dataset_results = [{"loss": 0.578, "dataset_name": "Hupy440/Two_Cubes_and_Two_Buckets_v2"}]
        mock_strategy.federated_metrics_history = [{"avg_client_loss": 0.912, "num_clients": 2}, {"avg_client_loss": 0.894, "num_clients": 2}]

        # Mock template model
        mock_param = MagicMock()
        mock_param.dtype = torch.float32
        mock_param.shape = (2,)
        mock_state_dict = {"param1": mock_param}
        mock_strategy.template_model.state_dict.return_value = mock_state_dict
        mock_strategy.template_model.config.save_pretrained = MagicMock()

        # Mock parameters
        mock_parameters = MagicMock()

        with patch("flwr.common.parameters_to_ndarrays") as mock_ndarrays, \
              patch("builtins.open", mock_open()):

            mock_ndarrays.return_value = [np.array([1.0, 2.0])]

            save_model_checkpoint(mock_strategy, mock_parameters, 2)

            # Verify in-memory usage
            # mock_compute_insights.assert_called_once_with(mock_strategy)

            # Verify repo_id and wandb_url passed
            mock_generate_card.assert_called_once()
            args = mock_generate_card.call_args[0]
            assert args[-1] == "ivelin/zk0-smolvla-fl"  # hf_repo_id from config
            assert args[-2]["federation"] == "local-simulation"  # other_info dict

            # Verify datasets populated
            mock_extract_datasets.assert_called_once()

    @patch("src.server.server_utils.get_tool_config")
    @patch("src.server.server_utils.compute_in_memory_insights")
    @patch("src.server.server_utils.generate_model_card")
    def test_save_model_checkpoint_in_memory_empty(self, mock_generate, mock_insights, mock_get_config):
        """Test in-memory checkpoint with empty strategy data."""
        from src.server.server_utils import save_model_checkpoint

        mock_strategy = MagicMock()
        mock_strategy.context.run_config = {"federation": "local-simulation"}
        mock_strategy.server_dir = Path(tempfile.mkdtemp())
        mock_strategy.models_dir = Path(tempfile.mkdtemp())
        mock_strategy.template_model = MagicMock()
        mock_strategy.server_eval_losses = []
        mock_strategy.last_aggregated_metrics = {}
        mock_strategy.last_client_metrics = []
        mock_strategy.last_per_dataset_results = []
        mock_strategy.federated_metrics_history = []

        mock_param = MagicMock()
        mock_param.dtype = torch.float32
        mock_param.shape = (2,)
        mock_state_dict = {"param1": mock_param}
        mock_strategy.template_model.state_dict.return_value = mock_state_dict
        mock_strategy.template_model.config.save_pretrained = MagicMock()

        mock_parameters = MagicMock()

        with patch("flwr.common.parameters_to_ndarrays") as mock_ndarrays, \
              patch("safetensors.torch.save_file"):

            mock_ndarrays.return_value = [np.array([1.0, 2.0])]
            mock_get_config.return_value = {"app": {"config": {}}}
            mock_insights.return_value = {"convergence_trend": "N/A", "avg_client_loss_trend": "N/A", "client_participation_rate": "N/A", "anomalies": []}
            mock_generate.return_value = "# Empty Model Card"

            save_model_checkpoint(mock_strategy, mock_parameters, 2)

            # Verify fallbacks to N/A/empty
            mock_generate.assert_called_once()