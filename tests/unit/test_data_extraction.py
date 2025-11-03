"""Unit tests for data extraction functions."""

from unittest.mock import MagicMock


class TestExtractTrainingHyperparameters:
    """Test extract_training_hyperparameters function."""

    def test_extract_training_hyperparameters(self):
        """Test hyperparameter extraction."""
        from src.server.model_utils import extract_training_hyperparameters

        mock_context = MagicMock()
        mock_context.run_config = {
            "num-server-rounds": 100,
            "fraction-fit": 0.8,
            "fraction-evaluate": 1.0,
            "eval-frequency": 2,
            "eval_batches": 8,
            "checkpoint_interval": 10,
        }

        pyproject_config = {
            "local-epochs": 15,
            "proximal_mu": 0.02,
            "initial_lr": 5e-4,
            "batch_size": 32,
            "dynamic_training_decay": True,
            "scheduler_type": "cosine_warm_restarts",
            "adaptive_lr_enabled": True,
            "adaptive_mu_enabled": True,
        }

        hyperparams = extract_training_hyperparameters(mock_context, pyproject_config)

        assert hyperparams["num_server_rounds"] == 100
        assert hyperparams["local_epochs"] == 15
        assert hyperparams["proximal_mu"] == 0.02
        assert hyperparams["initial_lr"] == 5e-4
        assert hyperparams["batch_size"] == 32
        assert hyperparams["fraction_fit"] == 0.8
        assert hyperparams["dynamic_training_decay"] is True
        assert hyperparams["scheduler_type"] == "cosine_warm_restarts"


class TestExtractDatasets:
    """Test extract_datasets function."""

    def test_extract_datasets_basic(self):
        """Test basic dataset extraction."""
        from src.server.model_utils import extract_datasets

        pyproject_config = {
            "clients": [
                {"name": "client1", "description": "desc1", "client_id": 0},
                {"name": "client2", "description": "desc2", "client_id": 1},
            ],
            "server": [
                {"name": "eval1", "description": "eval desc1", "evaldata_id": 0},
            ]
        }

        train_datasets, eval_datasets = extract_datasets(pyproject_config, is_simulation=False)

        assert len(train_datasets) == 2
        assert len(eval_datasets) == 1
        assert train_datasets[0]["name"] == "client1"
        assert eval_datasets[0]["name"] == "eval1"

    def test_extract_datasets_simulation_mode(self):
        """Test dataset extraction in simulation mode."""
        from src.server.model_utils import extract_datasets

        pyproject_config = {
            "clients": [
                {"name": "client1", "description": "desc1", "client_id": 0},
            ]
        }

        train_datasets, eval_datasets = extract_datasets(pyproject_config, is_simulation=True)

        assert len(train_datasets) == 2  # Original + simulation note
        assert train_datasets[1]["name"] == "Simulation Mode"
        assert "1 clients" in train_datasets[1]["description"]