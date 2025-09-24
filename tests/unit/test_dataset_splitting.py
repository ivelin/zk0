"""Real dataset tests for episode splitting - tests actual LeRobot dataset functionality."""

import pytest


# Import our functions
from src.client_app import get_all_client_configs

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
        # Load dataset metadata only (efficient)
        dataset = LeRobotDataset(repo_id=dataset_name)
        assert dataset.num_episodes > 0, f"Dataset {dataset_name} should have episodes"
        assert hasattr(dataset, 'num_frames'), f"Dataset {dataset_name} should have frame count"

    @pytest.mark.parametrize("dataset_name,n_eval", [
        ("lerobot/svla_so100_stacking", 3),
        ("lerobot/svla_so100_pickplace", 3),
        ("lerobot/svla_so100_sorting", 3),
        ("lerobot/svla_so101_pickplace", 3)
    ])
    def test_real_dataset_episode_splitting(self, dataset_name, n_eval):
        """Test episode splitting with real datasets."""
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

    def test_real_dataset_efficient_loading(self):
        """Test that efficient loading with episodes parameter works with real datasets."""
        dataset_name = "lerobot/svla_so100_stacking"

        # Load metadata first (efficient)
        metadata_dataset = LeRobotDataset(repo_id=dataset_name)
        total_episodes = metadata_dataset.num_episodes

        # Load with episode filtering (should be more efficient)
        n_eval = 3
        eval_episodes = list(range(total_episodes - n_eval, total_episodes))
        filtered_dataset = LeRobotDataset(repo_id=dataset_name, episodes=eval_episodes)

        # Verify we got the right episodes
        assert filtered_dataset.num_episodes == n_eval
        assert set(filtered_dataset.episodes) == set(eval_episodes)

    @pytest.mark.parametrize("dataset_name", [
        "lerobot/svla_so100_stacking",
        "lerobot/svla_so100_pickplace",
        "lerobot/svla_so100_sorting",
        "lerobot/svla_so101_pickplace"
    ])
    def test_real_dataset_frame_consistency(self, dataset_name):
        """Test that frame counts are consistent when splitting episodes."""
        # Load full dataset
        full_dataset = LeRobotDataset(repo_id=dataset_name)
        full_frames = full_dataset.num_frames

        # Split into train/eval
        total_episodes = full_dataset.num_episodes
        n_eval = min(3, total_episodes)  # Don't exceed available episodes

        train_episodes = list(range(total_episodes - n_eval))
        eval_episodes = list(range(total_episodes - n_eval, total_episodes))

        train_dataset = LeRobotDataset(repo_id=dataset_name, episodes=train_episodes)
        eval_dataset = LeRobotDataset(repo_id=dataset_name, episodes=eval_episodes)

        # Verify frame counts add up
        split_frames = train_dataset.num_frames + eval_dataset.num_frames
        assert split_frames == full_frames, \
            f"Frame count mismatch: full={full_frames}, split={split_frames}"

    def test_real_dataset_boundary_cases(self):
        """Test edge cases with real datasets."""
        dataset_name = "lerobot/svla_so100_stacking"

        # Load dataset and check its size
        dataset = LeRobotDataset(repo_id=dataset_name)
        total_episodes = dataset.num_episodes

        # Test case 1: Request more eval episodes than available
        n_eval_too_large = total_episodes + 5
        eval_episodes = list(range(max(0, total_episodes - n_eval_too_large), total_episodes))
        small_eval_dataset = LeRobotDataset(repo_id=dataset_name, episodes=eval_episodes)

        # Should get all available episodes
        assert small_eval_dataset.num_episodes == total_episodes
        assert set(small_eval_dataset.episodes) == set(range(total_episodes))

        # Test case 2: Request 0 eval episodes
        n_eval_zero = 0
        eval_episodes_zero = list(range(max(0, total_episodes - n_eval_zero), total_episodes))
        if eval_episodes_zero:  # Only create dataset if there are episodes to load
            zero_eval_dataset = LeRobotDataset(repo_id=dataset_name, episodes=eval_episodes_zero)
            # When n_eval=0, we should get all episodes for eval (not 0)
            assert zero_eval_dataset.num_episodes == total_episodes
        else:
            # If no episodes, skip this test case
            pytest.skip("Dataset too small for this test case")

    def test_real_dataset_episode_ordering(self):
        """Test that episode ordering is preserved when splitting."""
        dataset_name = "lerobot/svla_so100_stacking"

        dataset = LeRobotDataset(repo_id=dataset_name)
        total_episodes = dataset.num_episodes

        # Split into first half and second half
        mid_point = total_episodes // 2
        first_half_episodes = list(range(mid_point))
        second_half_episodes = list(range(mid_point, total_episodes))

        first_dataset = LeRobotDataset(repo_id=dataset_name, episodes=first_half_episodes)
        second_dataset = LeRobotDataset(repo_id=dataset_name, episodes=second_half_episodes)

        # Verify episode ordering
        assert first_dataset.episodes == first_half_episodes
        assert second_dataset.episodes == second_half_episodes

        # Verify no overlap
        assert set(first_dataset.episodes) & set(second_dataset.episodes) == set()



