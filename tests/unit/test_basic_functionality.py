"""Basic functionality tests to ensure test setup works correctly."""

import pytest
from unittest.mock import patch

from src.client_app import SmolVLAClient


@pytest.mark.unit
def test_client_can_be_imported():
    """Test that SmolVLAClient can be imported successfully."""
    assert SmolVLAClient is not None


@pytest.mark.unit
def test_client_initialization_basic(sample_client_config):
    """Test basic client initialization without complex dependencies."""
    with patch('src.task.get_model') as mock_get_model:

        mock_get_model.side_effect = Exception("Model loading disabled for test")

        # Should not raise exception
        client = SmolVLAClient(**sample_client_config)

        # Verify basic attributes are set
        assert hasattr(client, 'partition_id')
        assert hasattr(client, 'local_epochs')
        assert hasattr(client, 'device')
        assert hasattr(client, 'net')


@pytest.mark.unit
def test_flower_api_methods_exist(sample_client_config):
    """Test that all required Flower API methods exist."""
    with patch('src.task.get_model') as mock_get_model:
        mock_get_model.side_effect = Exception("Model loading disabled")

        client = SmolVLAClient(**sample_client_config)

        # Check that Flower API methods exist
        assert hasattr(client, 'get_parameters')
        assert hasattr(client, 'fit')
        assert hasattr(client, 'evaluate')

        # Check that methods are callable
        assert callable(client.get_parameters)
        assert callable(client.fit)
        assert callable(client.evaluate)

