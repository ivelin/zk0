"""Basic functionality tests to ensure test setup works correctly."""

import pytest
from unittest.mock import patch

from smolvla_example.client_app import SmolVLAClient


@pytest.mark.unit
def test_client_can_be_imported():
    """Test that SmolVLAClient can be imported successfully."""
    assert SmolVLAClient is not None


@pytest.mark.unit
def test_client_initialization_basic(sample_client_config):
    """Test basic client initialization without complex dependencies."""
    with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model, \
         patch('smolvla_example.client_app.AutoProcessor'):

        mock_model.from_pretrained.side_effect = Exception("Model loading disabled for test")

        # Should not raise exception
        client = SmolVLAClient(**sample_client_config)

        # Verify basic attributes are set
        assert hasattr(client, 'model_name')
        assert hasattr(client, 'device')
        assert hasattr(client, 'partition_id')
        assert hasattr(client, 'num_partitions')


@pytest.mark.unit
def test_flower_api_methods_exist(sample_client_config):
    """Test that all required Flower API methods exist."""
    with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model:
        mock_model.from_pretrained.side_effect = Exception("Model loading disabled")

        client = SmolVLAClient(**sample_client_config)

        # Check that Flower API methods exist
        assert hasattr(client, 'get_parameters')
        assert hasattr(client, 'set_parameters')
        assert hasattr(client, 'fit')
        assert hasattr(client, 'evaluate')

        # Check that methods are callable
        assert callable(client.get_parameters)
        assert callable(client.set_parameters)
        assert callable(client.fit)
        assert callable(client.evaluate)


@pytest.mark.unit
def test_client_configuration_storage(sample_client_config):
    """Test that client stores configuration correctly."""
    with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model:
        mock_model.from_pretrained.side_effect = Exception("Model loading disabled")

        client = SmolVLAClient(**sample_client_config)

        # Verify configuration is stored
        assert client.model_name == sample_client_config["model_name"]
        assert client.device == sample_client_config["device"]
        assert client.partition_id == sample_client_config["partition_id"]
        assert client.num_partitions == sample_client_config["num_partitions"]