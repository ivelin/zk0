"""Unit tests for dataset loading helper functions."""

import pytest
from unittest.mock import Mock, patch
from src.utils import load_lerobot_dataset, FilteredLeRobotDataset


class TestLoadLeRobotDataset:
    """Test the load_lerobot_dataset helper function."""

    @patch('src.utils.LEROBOT_AVAILABLE', True)
    @patch('src.utils.LeRobotDataset')
    def test_basic_loading(self, mock_lerobot_dataset):
        """Test basic dataset loading without filters."""
        # Mock the dataset
        mock_dataset = Mock()
        mock_dataset.num_episodes = 10
        mock_lerobot_dataset.return_value = mock_dataset

        # Call the function
        result = load_lerobot_dataset("test/repo", tolerance_s=0.001)

        # Assertions
        mock_lerobot_dataset.assert_called_once_with(
            repo_id="test/repo",
            tolerance_s=0.001
        )
        assert result == mock_dataset

    @patch('src.utils.LEROBOT_AVAILABLE', True)
    @patch('src.utils.LeRobotDataset')
    def test_episode_filter_last_n(self, mock_lerobot_dataset):
        """Test loading with last_n episode filter."""
        # Mock datasets
        mock_temp_dataset = Mock()
        mock_temp_dataset.num_episodes = 20
        mock_filtered_dataset = Mock()
        mock_filtered_dataset.num_episodes = 5

        mock_lerobot_dataset.side_effect = [mock_temp_dataset, mock_filtered_dataset]

        # Call the function
        result = load_lerobot_dataset(
            "test/repo",
            episode_filter={"last_n": 5}
        )

        # Assertions
        assert mock_lerobot_dataset.call_count == 2
        # First call for metadata
        mock_lerobot_dataset.assert_any_call(repo_id="test/repo", tolerance_s=0.0001)
        # Second call with episodes
        mock_lerobot_dataset.assert_any_call(
            repo_id="test/repo",
            episodes=[15, 16, 17, 18, 19],
            tolerance_s=0.0001
        )
        assert result == mock_filtered_dataset

    @patch('src.utils.LEROBOT_AVAILABLE', True)
    @patch('src.utils.LeRobotDataset')
    def test_episode_filter_exclude_last_n(self, mock_lerobot_dataset):
        """Test loading with exclude_last_n episode filter."""
        # Mock datasets
        mock_temp_dataset = Mock()
        mock_temp_dataset.num_episodes = 20
        mock_filtered_dataset = Mock()
        mock_filtered_dataset.num_episodes = 15

        mock_lerobot_dataset.side_effect = [mock_temp_dataset, mock_filtered_dataset]

        # Call the function
        result = load_lerobot_dataset(
            "test/repo",
            episode_filter={"exclude_last_n": 5}
        )

        # Assertions
        assert mock_lerobot_dataset.call_count == 2
        # Second call with episodes
        mock_lerobot_dataset.assert_any_call(
            repo_id="test/repo",
            episodes=list(range(15)),
            tolerance_s=0.0001
        )
        assert result == mock_filtered_dataset

    @patch('src.utils.LEROBOT_AVAILABLE', True)
    @patch('src.utils.LeRobotDataset')
    @patch('src.utils.FilteredLeRobotDataset')
    def test_delta_timestamps(self, mock_filtered_dataset, mock_lerobot_dataset):
        """Test loading with delta timestamps."""
        # Mock datasets
        mock_dataset = Mock()
        mock_dataset.num_episodes = 10
        mock_lerobot_dataset.return_value = mock_dataset

        mock_filtered = Mock()
        mock_filtered_dataset.return_value = mock_filtered

        delta_timestamps = {"observation.image": [-0.1, 0.0]}

        # Call the function
        result = load_lerobot_dataset(
            "test/repo",
            delta_timestamps=delta_timestamps
        )

        # Assertions
        mock_filtered_dataset.assert_called_once()
        assert result == mock_filtered

    @patch('src.utils.LEROBOT_AVAILABLE', True)
    @patch('src.client_app.load_config')
    @patch('src.client_app.get_episode_split')
    @patch('src.utils.LeRobotDataset')
    def test_split_parameter(self, mock_lerobot_dataset, mock_get_episode_split, mock_load_config):
        """Test loading with split parameter."""
        # Mock config loading
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        mock_get_episode_split.return_value = {"last_n": 3}

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.num_episodes = 10
        mock_lerobot_dataset.side_effect = [mock_dataset, mock_dataset]  # Temp and filtered

        # Call the function
        result = load_lerobot_dataset("test/repo", split="eval")

        # Assertions
        mock_load_config.assert_called_once_with("src/configs/default.yaml")
        mock_get_episode_split.assert_called_once_with(mock_config, "eval")
        assert result == mock_dataset

    @patch('src.utils.LEROBOT_AVAILABLE', False)
    def test_lerobot_not_available(self):
        """Test error when LeRobot is not available."""
        with pytest.raises(RuntimeError, match="LeRobot not available"):
            load_lerobot_dataset("test/repo")

    @patch('src.utils.LEROBOT_AVAILABLE', True)
    @patch('src.utils.LeRobotDataset')
    def test_loading_failure(self, mock_lerobot_dataset):
        """Test handling of loading failures."""
        mock_lerobot_dataset.side_effect = Exception("Load failed")

        with pytest.raises(RuntimeError, match="Dataset loading failed"):
            load_lerobot_dataset("test/repo")


class TestFilteredLeRobotDataset:
    """Test the FilteredLeRobotDataset class."""

    @patch('src.utils.LEROBOT_AVAILABLE', True)
    @patch('src.utils.calculate_episode_data_index')
    @patch('src.utils.load_stats')
    @patch('src.utils.load_info')
    def test_initialization(self, mock_load_info, mock_load_stats, mock_calc_index):
        """Test FilteredLeRobotDataset initialization."""
        # Skip if FilteredLeRobotDataset is not available
        from src.utils import FilteredLeRobotDataset
        if FilteredLeRobotDataset is None:
            pytest.skip("FilteredLeRobotDataset not available in test environment")

        # Mocks
        mock_hf_dataset = Mock()
        mock_load_stats.return_value = {"test": "stats"}
        mock_load_info.return_value = {"fps": 30, "features": {}, "tasks": {}}
        mock_calc_index.return_value = [0, 1, 2]

        # Create instance
        dataset = FilteredLeRobotDataset(
            repo_id="test/repo",
            hf_dataset=mock_hf_dataset,
            delta_timestamps={"test": [0.0]}
        )

        # Assertions
        assert dataset.repo_id == "test/repo"
        assert dataset.hf_dataset == mock_hf_dataset
        mock_load_stats.assert_called_once_with("test/repo")
        mock_load_info.assert_called_once_with("test/repo")

    @patch('src.utils.LEROBOT_AVAILABLE', True)
    @patch('src.utils.calculate_episode_data_index')
    @patch('src.utils.load_stats')
    @patch('src.utils.load_info')
    @patch('src.utils.check_delta_timestamps')
    @patch('src.utils.get_delta_indices')
    def test_delta_indices_computation(self, mock_get_indices, mock_check_timestamps,
                                      mock_load_info, mock_load_stats, mock_calc_index):
        """Test delta indices computation."""
        # Skip if FilteredLeRobotDataset is not available
        from src.utils import FilteredLeRobotDataset
        if FilteredLeRobotDataset is None:
            pytest.skip("FilteredLeRobotDataset not available in test environment")

        # Mocks
        mock_hf_dataset = Mock()
        mock_load_stats.return_value = {"test": "stats"}
        mock_load_info.return_value = {"fps": 30, "features": {}, "tasks": {}}
        mock_calc_index.return_value = [0, 1, 2]
        mock_get_indices.return_value = [10, 20]

        delta_timestamps = {"observation.image": [-0.1, 0.0]}

        # Create instance
        dataset = FilteredLeRobotDataset(
            repo_id="test/repo",
            hf_dataset=mock_hf_dataset,
            delta_timestamps=delta_timestamps
        )

        # Assertions
        mock_check_timestamps.assert_called_once_with(delta_timestamps, 30, None)
        mock_get_indices.assert_called_once_with(delta_timestamps, 30)
        assert dataset.delta_indices == [10, 20]