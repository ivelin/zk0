"""Pytest configuration and fixtures for SmolVLA tests."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np

# Add the smolvla_example directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "smolvla_example"))


@pytest.fixture
def sample_client_config():
    """Sample client configuration for testing."""
    return {
        "model_name": "lerobot/smolvla_base",
        "device": "cpu",
        "partition_id": 0,
        "num_partitions": 2
    }


@pytest.fixture
def client_config():
    """Default client configuration for integration tests."""
    return {
        "model_name": "lerobot/smolvla_base",
        "device": "cpu",
        "partition_id": 0,
        "num_partitions": 2
    }


@pytest.fixture
def mock_model():
    """Simple mock model for Flower API testing."""
    model = MagicMock()

    # Mock state_dict to return numpy arrays (what Flower expects)
    model.state_dict.return_value = {
        'param1': np.array([1.0, 2.0, 3.0]),
        'param2': np.array([4.0, 5.0, 6.0])
    }
    model.load_state_dict = MagicMock()

    # Mock training methods
    model.train = MagicMock()
    model.eval = MagicMock()

    return model


@pytest.fixture
def mock_optimizer():
    """Mock optimizer for testing."""
    optimizer = MagicMock()
    optimizer.step = MagicMock()
    optimizer.zero_grad = MagicMock()
    optimizer.state_dict.return_value = {'lr': 0.01}
    optimizer.param_groups = [{'lr': 0.01}]
    return optimizer


@pytest.fixture
def mock_processor():
    """Mock processor for transformers."""
    processor = MagicMock()
    return processor


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for tests."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(scope="session")
def test_logger():
    """Test logger fixture."""
    import logging
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    return logger