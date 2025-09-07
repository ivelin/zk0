"""Pytest configuration and fixtures for SmolVLA tests."""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import torch

# Add the smolvla_example directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "smolvla_example"))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "skip_in_ci: mark test to skip in CI environment")


def pytest_runtest_setup(item):
    """Skip tests marked with skip_in_ci in CI environment."""
    if item.get_closest_marker("skip_in_ci") and os.getenv("CI"):
        pytest.skip("Test skipped in CI environment")


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def cached_smolvla_model():
    """Session-scoped fixture that provides a mock SmolVLA model for testing."""
    from unittest.mock import MagicMock
    import numpy as np

    # Create a mock model that behaves like SmolVLA
    model = MagicMock()

    # Mock state_dict to return numpy arrays (what Flower expects)
    model.state_dict.return_value = {
        'vision_encoder.weight': np.random.randn(768, 1024).astype(np.float32),
        'language_model.embed_tokens.weight': np.random.randn(32000, 768).astype(np.float32),
        'action_head.weight': np.random.randn(14, 768).astype(np.float32),
    }

    # Mock methods
    model.load_state_dict = MagicMock()
    model.train = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)

    # Mock parameters to return proper tensor-like objects
    mock_param = MagicMock()
    mock_param.requires_grad = True
    mock_param.data = MagicMock()
    model.parameters = MagicMock(return_value=[mock_param])

    # Mock vision_encoder for parameter freezing
    vision_encoder = MagicMock()
    vision_encoder.parameters = MagicMock(return_value=[MagicMock()])
    model.vision_encoder = vision_encoder

    # Mock forward pass
    model.return_value = MagicMock()
    model.return_value.loss = MagicMock()
    model.return_value.loss.item.return_value = 0.5

    print("Mock SmolVLA model created for test session")
    return model


@pytest.fixture(scope="session")
def cached_so100_dataset():
    """Session-scoped fixture that provides a mock SO-100 dataset for testing."""
    from unittest.mock import MagicMock

    # Create a mock dataset that behaves like LeRobotDataset
    dataset = MagicMock()

    # Mock dataset properties
    dataset.num_episodes = 10
    dataset.num_frames = 100

    # Mock episode selection
    mock_episode = MagicMock()
    mock_episode.__len__ = MagicMock(return_value=20)
    dataset.select_episodes = MagicMock(return_value=mock_episode)

    # Mock data access
    dataset.__getitem__ = MagicMock(return_value={
        'observation.image': torch.randn(3, 224, 224),
        'observation.state': torch.randn(7),
        'action': torch.randn(14),
        'timestamp': torch.tensor(0.0),
    })

    print("Mock SO-100 dataset created for test session")
    return dataset


@pytest.fixture(scope="session")
def preloaded_client(cached_smolvla_model, cached_so100_dataset, sample_client_config):
    """Client fixture with preloaded model and dataset for faster tests."""
    from src.client_app import SmolVLAClient

    # Create client without calling __init__ methods
    client = object.__new__(SmolVLAClient)

    # Set basic attributes
    client.config = None
    client.model_name = sample_client_config["model_name"]
    client.device = sample_client_config["device"]
    client.partition_id = sample_client_config["partition_id"]
    client.num_partitions = sample_client_config["num_partitions"]
    client.model = cached_smolvla_model
    client.processor = None
    client.federated_dataset = None
    client.train_loader = None
    client.val_loader = None

    # Setup logging
    import logging
    client.logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Create output directory
    from pathlib import Path
    client.output_dir = Path("outputs") / "smolvla_federated" / f"client_{client.partition_id}"
    client.output_dir.mkdir(parents=True, exist_ok=True)

    # Create optimizer if model is available
    if client.model is not None:
        client.optimizer = torch.optim.Adam(
            [p for p in client.model.parameters() if p.requires_grad],
            lr=1e-4
        )

        # Create simple mock dataloader for testing
        if cached_so100_dataset is not None:
            try:
                # Create a simple mock dataloader
                mock_batch = {
                    'observation.image': torch.randn(4, 3, 224, 224),
                    'observation.state': torch.randn(4, 7),
                    'action': torch.randn(4, 14),
                    'input_ids': torch.randint(0, 32000, (4, 128)),
                    'labels': torch.randint(0, 32000, (4, 128)),
                }

                from unittest.mock import MagicMock
                mock_dataloader = MagicMock()
                mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch] * 2))  # Reduced for faster tests
                client.train_loader = mock_dataloader
                print("Created mock test dataloader")
            except Exception as e:
                print(f"Failed to create mock dataloader: {e}")
                client.train_loader = None
    else:
        client.optimizer = None

    return client