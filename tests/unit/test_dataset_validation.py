"""Real dataset validation tests - tests actual LeRobot dataset validation and time synchronization."""

import pytest
import numpy as np
import torch
import logging

# Import the validation functions from LeRobot
from lerobot.datasets.utils import check_timestamps_sync, check_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Import our configuration functions
from src.client_app import load_datasets_config, get_client_dataset_config, get_episode_split


@pytest.fixture
def datasets_config():
    """Load the datasets configuration."""
    clients = load_datasets_config()
    return type('Config', (), {'clients': clients})()


class TestDatasetValidation:
    """Test dataset validation with detailed debug output."""

    def setup_method(self):
        """Set up logging for detailed test output."""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)

    def test_datasets_config_loading(self, datasets_config):
        """Test that datasets configuration loads correctly."""
        assert hasattr(datasets_config, 'clients')
        assert len(datasets_config.clients) == 4

        # Check each client has required fields
        for client in datasets_config.clients:
            assert hasattr(client, 'name')
            assert hasattr(client, 'client_id')
            assert hasattr(client, 'tolerance_s')
            assert hasattr(client, 'fps')
            assert hasattr(client, 'last_n_episodes_for_eval')

            # Validate tolerance_s is reasonable (should be small, around 1/fps)
            assert 1e-6 <= client.tolerance_s <= 1e-2, f"Tolerance {client.tolerance_s} seems unreasonable"

    def test_client_config_mapping(self, datasets_config):
        """Test that client configurations map correctly to client IDs."""
        for i in range(4):
            config = get_client_dataset_config(i)
            assert config.client_id == i
            assert "lerobot/svla_so" in config.name  # Handle both SO-100 and SO-101

    def test_episode_split_logic(self, datasets_config):
        """Test episode splitting logic for train/eval."""
        for client_config in datasets_config.clients:
            # Test eval split
            eval_split = get_episode_split(client_config, "eval")
            assert "last_n" in eval_split
            assert eval_split["last_n"] == client_config.last_n_episodes_for_eval

            # Test train split
            train_split = get_episode_split(client_config, "train")
            assert "exclude_last_n" in train_split
            assert train_split["exclude_last_n"] == client_config.last_n_episodes_for_eval

    @pytest.mark.parametrize("client_id", [0, 1, 2, 3])
    def test_tolerance_values_reasonable(self, client_id):
        """Test that tolerance values are reasonable for the given FPS."""
        config = get_client_dataset_config(client_id)
        expected_tolerance = 1.0 / config.fps  # 1/fps

        # Tolerance should be close to 1/fps, but can be smaller for precision
        assert config.tolerance_s <= expected_tolerance * 2, \
            f"Tolerance {config.tolerance_s} too large for fps {config.fps}"

        # But not too small (would be too strict)
        assert config.tolerance_s >= expected_tolerance / 1000, \
            f"Tolerance {config.tolerance_s} too small for fps {config.fps}"


    def test_timestamp_validation_with_noise(self):
        """Test timestamp validation with realistic noise."""
        fps = 30
        proper_tolerance = 1e-4
        n_timestamps = 100

        # Create timestamps with small random noise (realistic scenario)
        base_timestamps = np.arange(n_timestamps) / fps
        noise = np.random.normal(0, 1e-5, n_timestamps)  # Small noise
        timestamps = base_timestamps + noise

        episode_indices = np.zeros(n_timestamps, dtype=int)
        episode_data_index = {"to": torch.LongTensor([n_timestamps])}

        # Should pass with proper tolerance
        result = check_timestamps_sync(
            timestamps, episode_indices, episode_data_index,
            fps, proper_tolerance, raise_value_error=False
        )
        assert result, f"Timestamps with small noise should pass. Timestamps: {timestamps[:5]}"

    def test_delta_timestamps_validation(self):
        """Test delta timestamps validation."""
        fps = 30
        proper_tolerance = 1e-4

        # Valid delta timestamps (multiples of 1/fps)
        valid_deltas = {
            "action": [0.0, 1/30, 2/30, 3/30],  # Exact multiples
            "observation": [0.0, 1/30]  # Exact multiples
        }

        result = check_delta_timestamps(valid_deltas, fps, proper_tolerance, raise_value_error=False)
        assert result, "Valid delta timestamps should pass"

        # Invalid delta timestamps (not multiples of 1/fps)
        invalid_deltas = {
            "action": [0.0, 0.015, 0.025],  # Not exact multiples of 1/30
        }

        result = check_delta_timestamps(invalid_deltas, fps, proper_tolerance, raise_value_error=False)
        assert not result, "Invalid delta timestamps should fail"

    def test_dataset_config_consistency(self, datasets_config):
        """Test that all datasets have consistent configuration."""
        # Test all datasets from config
        for client in datasets_config.clients:
            dataset_name = client.name

            # All datasets should have same tolerance and fps for consistency
            assert client.tolerance_s == 0.0001, f"Dataset {dataset_name} has wrong tolerance_s: {client.tolerance_s}"
            assert client.fps == 30, f"Dataset {dataset_name} has wrong fps: {client.fps}"
            assert client.last_n_episodes_for_eval == 3, f"Dataset {dataset_name} has wrong last_n_episodes_for_eval: {client.last_n_episodes_for_eval}"





