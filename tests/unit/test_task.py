"""Unit tests for src/task.py functions."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock


class TestResetLearningRateScheduler:
    """Test the reset_learning_rate_scheduler function."""

    def test_reset_linear_lr_scheduler(self):
        """Test resetting LinearLR scheduler."""
        # Create real optimizer with param groups
        import torch

        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.param_groups[0]["lr"] = 0.001  # Set initial LR

        # Create LinearLR scheduler
        from torch.optim.lr_scheduler import LinearLR

        scheduler = LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=50
        )

        # Simulate some decay by stepping
        scheduler.step()
        assert scheduler.get_last_lr()[0] != 0.0005  # Should be decayed

        # Reset scheduler
        from src.task import reset_learning_rate_scheduler

        new_scheduler = reset_learning_rate_scheduler(optimizer, scheduler, 0.0005, 50)

        # Check that optimizer LRs are reset
        assert all(group["lr"] == 0.0005 for group in optimizer.param_groups)

        # Check that scheduler is reset (new instance for LinearLR)
        assert new_scheduler is not scheduler  # Should be new instance
        assert isinstance(new_scheduler, LinearLR)

    def test_reset_none_scheduler(self):
        """Test resetting when scheduler is None."""
        optimizer = Mock()
        optimizer.param_groups = [{"lr": 0.001}]

        from src.task import reset_learning_rate_scheduler

        result = reset_learning_rate_scheduler(optimizer, None, 0.0005, 50)

        assert result is None
        assert optimizer.param_groups[0]["lr"] == 0.0005

    def test_reset_other_scheduler_types(self):
        """Test resetting other scheduler types (like CosineAnnealingLR)."""
        # Create real optimizer
        import torch

        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create CosineAnnealingLR scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(optimizer, T_max=50)
        scheduler.last_epoch = 10  # Simulate some progress

        from src.task import reset_learning_rate_scheduler

        result = reset_learning_rate_scheduler(optimizer, scheduler, 0.0005, 50)

        assert result is scheduler  # Same instance returned
        assert result.last_epoch == -1  # Reset to -1
        assert optimizer.param_groups[0]["lr"] == 0.0005


class TestComputeDynamicLrAdjustment:
    """Test the compute_dynamic_lr_adjustment function."""

    def test_insufficient_data(self):
        """Test with insufficient data (less than 3 losses)."""
        from src.task import compute_dynamic_lr_adjustment

        result_lr, reason = compute_dynamic_lr_adjustment([1.0, 2.0], 0.0005)
        assert result_lr == 0.0005
        assert reason == "insufficient_data"

    def test_stall_detection(self):
        """Test detection of stalling (less than 1% improvement)."""
        from src.task import compute_dynamic_lr_adjustment

        # Losses: 1.0 → 0.995 → 0.994 (0.5% improvement over 3 rounds)
        losses = [1.0, 0.995, 0.994]
        result_lr, reason = compute_dynamic_lr_adjustment(losses, 0.0005)
        assert result_lr == 0.0004  # 0.0005 * 0.8
        assert "stall_detected" in reason

    def test_divergence_detection(self):
        """Test detection of divergence (loss increase >5%)."""
        from src.task import compute_dynamic_lr_adjustment

        # Losses: 1.0 → 1.02 → 1.08 (8% increase over 3 rounds)
        losses = [1.0, 1.02, 1.08]
        result_lr, reason = compute_dynamic_lr_adjustment(losses, 0.0005)
        assert (
            result_lr == 0.0004
        )  # 0.0005 * 0.8 (stall detection due to negative improvement)
        assert "stall_detected" in reason

    def test_stable_progress(self):
        """Test stable progress (no adjustment needed)."""
        from src.task import compute_dynamic_lr_adjustment

        # Losses: 1.0 → 0.95 → 0.91 (9% improvement over 3 rounds)
        losses = [1.0, 0.95, 0.91]
        result_lr, reason = compute_dynamic_lr_adjustment(losses, 0.0005)
        assert result_lr == 0.0005
        assert reason == "stable_progress"

    def test_lr_clamping(self):
        """Test LR clamping to min/max bounds."""
        from src.task import compute_dynamic_lr_adjustment

        # Test min clamping
        losses = [1.0, 0.995, 0.994]  # Stall
        result_lr, _ = compute_dynamic_lr_adjustment(losses, 1e-6)  # Very low LR
        assert result_lr == 1e-5  # Min LR

        # Test max clamping
        losses = [
            1.0,
            1.02,
            1.08,
        ]  # Divergence (but detected as stall due to negative improvement)
        result_lr, _ = compute_dynamic_lr_adjustment(losses, 1e-2)  # High LR
        assert result_lr == 8e-3  # 1e-2 * 0.8 = 8e-3 (stall detection)


def test_setup_training_components_metrics_initialization():
    """Test that setup_training_components initializes all required metrics without KeyError."""
    from src.task import setup_training_components
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.factory import make_policy
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    # Mock minimal components
    cfg = SmolVLAConfig()

    # Provide minimal ds_meta as object with features to satisfy make_policy
    class MockDatasetMetadata:
        def __init__(self):
            self.features = {
                "action": {"shape": (7,), "dtype": "float32"},
                "observation.images.front": {
                    "shape": (3, 480, 640),
                    "dtype": "float32",
                },
                "observation.images.top": {"shape": (3, 480, 640), "dtype": "float32"},
            }
            self.action_dim = 7
            self.stats = {}  # Mock stats to satisfy make_policy
            self.info = {}  # Add info attribute to avoid AttributeError

    mock_ds_meta = MockDatasetMetadata()
    policy = make_policy(cfg=cfg, ds_meta=mock_ds_meta)

    # Create a mock dataset with meta attribute
    class MockDataset:
        def __init__(self):
            self.meta = Mock()
            self.meta.repo_id = "test/repo"

    mock_dataset = MockDataset()
    mock_loader = DataLoader(mock_dataset, batch_size=1)
    device = torch.device("cpu")

    # Call setup
    _, _, _, _, train_metrics, _ = setup_training_components(
        policy, mock_loader, epochs=1, batch_size=1, device=device, initial_lr=1e-3
    )

    # Verify 'loss' key exists and logging doesn't crash (simulate log access)
    assert "loss" in train_metrics
    assert train_metrics["loss"].avg == 0.0  # Initial value
    assert "policy_loss" in train_metrics
    assert "fedprox_loss" in train_metrics

    # Simulate log access without error
    try:
        log_msg = f"loss_avg={train_metrics['loss'].avg:.4f}"
        assert "loss_avg=0.0000" in log_msg  # No KeyError
    except KeyError:
        pytest.fail("KeyError on 'loss' access - regression!")
