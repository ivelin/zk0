"""Unit tests for server_app.py functions."""

import pytest
from src.server_app import check_early_stopping, update_early_stopping_tracking


class MockStrategy:
    """Mock strategy class for testing early stopping functions."""

    def __init__(self, patience=10):
        self.early_stopping_patience = patience
        self.best_eval_loss = float('inf')
        self.rounds_without_improvement = 0
        self.early_stopping_triggered = False


def test_check_early_stopping_disabled():
    """Test early stopping when patience is 0 (disabled)."""
    should_stop, new_rounds = check_early_stopping(1.0, 2.0, 0, 0)
    assert should_stop is False
    assert new_rounds == 0


def test_check_early_stopping_improvement():
    """Test early stopping when loss improves."""
    should_stop, new_rounds = check_early_stopping(0.5, 1.0, 5, 10)
    assert should_stop is False
    assert new_rounds == 0


def test_check_early_stopping_no_improvement():
    """Test early stopping when loss doesn't improve."""
    should_stop, new_rounds = check_early_stopping(1.0, 0.5, 5, 10)
    assert should_stop is False
    assert new_rounds == 6


def test_check_early_stopping_trigger():
    """Test early stopping when patience is exceeded."""
    should_stop, new_rounds = check_early_stopping(1.0, 0.5, 9, 10)
    assert should_stop is True
    assert new_rounds == 10


def test_update_early_stopping_tracking_improvement():
    """Test updating early stopping tracking when loss improves."""
    strategy = MockStrategy(patience=5)
    strategy.best_eval_loss = 1.0

    update_early_stopping_tracking(strategy, 3, 0.8)

    assert strategy.best_eval_loss == 0.8
    assert strategy.rounds_without_improvement == 0
    assert strategy.early_stopping_triggered is False


def test_update_early_stopping_tracking_no_improvement():
    """Test updating early stopping tracking when loss doesn't improve."""
    strategy = MockStrategy(patience=5)
    strategy.best_eval_loss = 0.8
    strategy.rounds_without_improvement = 2

    update_early_stopping_tracking(strategy, 4, 1.0)

    assert strategy.best_eval_loss == 0.8
    assert strategy.rounds_without_improvement == 3
    assert strategy.early_stopping_triggered is False


def test_update_early_stopping_tracking_trigger():
    """Test updating early stopping tracking when patience is exceeded."""
    strategy = MockStrategy(patience=3)
    strategy.best_eval_loss = 0.8
    strategy.rounds_without_improvement = 2

    update_early_stopping_tracking(strategy, 5, 1.0)

    assert strategy.best_eval_loss == 0.8
    assert strategy.rounds_without_improvement == 3
    assert strategy.early_stopping_triggered is True


def test_update_early_stopping_tracking_already_triggered():
    """Test that tracking is skipped when early stopping is already triggered."""
    strategy = MockStrategy(patience=5)
    strategy.early_stopping_triggered = True
    strategy.best_eval_loss = 0.8
    strategy.rounds_without_improvement = 3

    update_early_stopping_tracking(strategy, 6, 1.0)

    # Values should remain unchanged
    assert strategy.best_eval_loss == 0.8
    assert strategy.rounds_without_improvement == 3
    assert strategy.early_stopping_triggered is True