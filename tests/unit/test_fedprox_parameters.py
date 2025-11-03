"""Unit tests for FedProx parameter computation."""

import pytest
from src.server.strategy import AggregateEvaluationStrategy


class TestComputeFedproxParameters:
    """Test the compute_fedprox_parameters method."""

    def test_fixed_adjustment_disabled(self):
        """Test fixed adjustment when dynamic_training_decay is disabled."""

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

        # Create test instance without current_lr set
        test_strategy = AggregateEvaluationStrategy.__new__(AggregateEvaluationStrategy)
        test_strategy.proximal_mu = 0.01
        # Don't set current_lr - should be initialized from config

        app_config = {"dynamic_training_decay": False, "initial_lr": 0.001}

        current_mu, current_lr = test_strategy.compute_fedprox_parameters(1, app_config)

        assert current_mu == 0.01
        assert current_lr == 0.001  # Should be initialized from config
        assert test_strategy.current_lr == 0.001  # Should be set for future rounds