"""Real dataset tests for episode splitting - tests actual LeRobot dataset functionality."""

import os
import pytest


# Import our functions

# Import LeRobot for real dataset testing
from lerobot.datasets.lerobot_dataset import LeRobotDataset

















class TestRealDatasetSplitting:
    """Test dataset splitting with actual LeRobot datasets."""


    @pytest.mark.parametrize("dataset_name", [
        "lerobot/svla_so100_stacking",
        "lerobot/svla_so100_pickplace",
        "lerobot/svla_so100_sorting"
    ])
    def test_real_dataset_episode_count(self, dataset_name):
        """Test that we can load real datasets and get correct episode counts."""
        try:
            # Load dataset metadata only (efficient)
            dataset = LeRobotDataset(repo_id=dataset_name)
            assert dataset.num_episodes > 0, f"Dataset {dataset_name} should have episodes"
            assert hasattr(dataset, 'num_frames'), f"Dataset {dataset_name} should have frame count"
        except ValueError as e:
            if "timestamps unexpectedly violate the tolerance" in str(e):
                pytest.skip(f"Dataset {dataset_name} has synchronization issues during data collection - skipping")
            else:
                raise

    @pytest.mark.parametrize("dataset_name,n_eval", [
        ("lerobot/svla_so100_stacking", 3),
        ("lerobot/svla_so100_pickplace", 3),
        ("lerobot/svla_so100_sorting", 3),
        ("lerobot/svla_so101_pickplace", 3)
    ])
    def test_real_dataset_episode_splitting(self, dataset_name, n_eval):
        """Test episode splitting with real datasets."""
        try:
            # Load full dataset first to get total count
            full_dataset = LeRobotDataset(repo_id=dataset_name)
            total_episodes = full_dataset.num_episodes

            # Calculate expected episode ranges
            expected_train_episodes = list(range(total_episodes - n_eval))
            expected_eval_episodes = list(range(total_episodes - n_eval, total_episodes))

            # Load datasets with episode filtering
            train_dataset = LeRobotDataset(repo_id=dataset_name, episodes=expected_train_episodes)
            eval_dataset = LeRobotDataset(repo_id=dataset_name, episodes=expected_eval_episodes)

            # Verify episode counts
            assert train_dataset.num_episodes == len(expected_train_episodes)
            assert eval_dataset.num_episodes == len(expected_eval_episodes)
            assert train_dataset.num_episodes + eval_dataset.num_episodes == total_episodes

            # Verify no episode overlap by checking episode indices
            # Note: LeRobotDataset.episodes contains the actual episode indices
            train_episode_indices = set(train_dataset.episodes)
            eval_episode_indices = set(eval_dataset.episodes)

            # Ensure no overlap
            overlap = train_episode_indices & eval_episode_indices
            assert len(overlap) == 0, f"Data leakage detected: episodes {overlap} in both train and eval sets"

            # Verify all episodes are accounted for
            all_episode_indices = train_episode_indices | eval_episode_indices
            expected_all_indices = set(range(total_episodes))
            assert all_episode_indices == expected_all_indices, f"Missing episodes: {expected_all_indices - all_episode_indices}"
        except ValueError as e:
            if "timestamps unexpectedly violate the tolerance" in str(e):
                pytest.skip(f"Dataset {dataset_name} has synchronization issues during data collection - skipping")
            else:
                raise







