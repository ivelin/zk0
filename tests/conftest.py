"""Pytest configuration and fixtures for SmolVLA tests."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

# Add the smolvla_example directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "smolvla_example"))


@pytest.fixture
def sample_client_config():
    """Sample client configuration for testing."""
    from unittest.mock import MagicMock
    mock_trainloader = MagicMock()
    mock_trainloader.dataset.meta.repo_id = "test/repo"

    return {
        "partition_id": 0,
        "local_epochs": 1,
        "trainloader": mock_trainloader,
        "nn_device": "cpu",
        "dataset_repo_id": "test/repo"
    }


@pytest.fixture
def client_config():
    """Default client configuration for integration tests."""
    from unittest.mock import MagicMock
    mock_trainloader = MagicMock()
    mock_trainloader.dataset.meta.repo_id = "test/repo"

    return {
        "partition_id": 0,
        "local_epochs": 1,
        "trainloader": mock_trainloader,
        "nn_device": "cpu",
        "dataset_repo_id": "test/repo"
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