"""Unit tests for SmolVLAClient class - focused on Flower API integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from smolvla_example.client_app import SmolVLAClient, get_device


@pytest.mark.unit
class TestGetDevice:
    """Test cases for get_device function."""

    @patch('torch.cuda.is_available', return_value=True)
    def test_get_device_auto_with_cuda(self, mock_cuda_available):
        """Test get_device with auto when CUDA is available."""
        result = get_device("auto")
        assert result == "cuda"

    @patch('torch.cuda.is_available', return_value=False)
    def test_get_device_auto_without_cuda(self, mock_cuda_available):
        """Test get_device with auto when CUDA is not available."""
        result = get_device("auto")
        assert result == "cpu"

    def test_get_device_cpu(self):
        """Test get_device with explicit cpu."""
        result = get_device("cpu")
        assert result == "cpu"

    def test_get_device_cuda(self):
        """Test get_device with explicit cuda."""
        result = get_device("cuda")
        assert result == "cuda"


@pytest.mark.unit
class TestSmolVLAClientFlowerAPI:
    """Test Flower API integration for SmolVLAClient."""

    @pytest.fixture
    def mock_client_with_model(self, sample_client_config, mock_model, mock_optimizer):
        """Create a client with mocked model for Flower API testing."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('smolvla_example.client_app.AutoProcessor'), \
             patch('smolvla_example.client_app.torch.optim.Adam', return_value=mock_optimizer):

            mock_model_class.from_pretrained.return_value = mock_model

            # Create client with mocked dependencies
            client = SmolVLAClient(**sample_client_config)
            client.model = mock_model
            client.optimizer = mock_optimizer
            return client

    def test_get_parameters_success(self, mock_client_with_model):
        """Test get_parameters returns Flower-compatible format."""
        try:
            from flwr.common import GetParametersIns
        except ImportError:
            pytest.skip("Flower not installed")

        result = mock_client_with_model.get_parameters(GetParametersIns(config={}))

        # Verify Flower API contract
        assert hasattr(result, 'parameters')
        assert hasattr(result.parameters, 'tensors')
        assert hasattr(result, 'status')
        assert result.status.code.value == 0  # OK

        # Verify tensors are numpy arrays (Flower requirement)
        assert isinstance(result.parameters.tensors, list)
        for tensor in result.parameters.tensors:
            assert isinstance(tensor, np.ndarray)

    def test_get_parameters_no_model(self, sample_client_config):
        """Test get_parameters when model is not available."""
        try:
            from flwr.common import GetParametersIns
        except ImportError:
            pytest.skip("Flower not installed")

        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**sample_client_config)

            result = client.get_parameters(GetParametersIns(config={}))

            # Should return empty parameters gracefully
            assert result.status.code.value == 0  # OK
            assert len(result.parameters.tensors) == 0

    def test_set_parameters_success(self, mock_client_with_model):
        """Test set_parameters accepts Flower-compatible format."""
        # Flower sends parameters as list of numpy arrays
        parameters = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0])
        ]

        # Should not raise exception
        mock_client_with_model.set_parameters(parameters)

        # Verify model.load_state_dict was called
        mock_client_with_model.model.load_state_dict.assert_called_once()

    def test_set_parameters_no_model(self, sample_client_config):
        """Test set_parameters when model is not available."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**sample_client_config)

            parameters = [np.array([1.0, 2.0, 3.0])]

            # Should not raise exception
            client.set_parameters(parameters)

    def test_fit_basic_workflow(self, mock_client_with_model):
        """Test fit method with basic Flower workflow."""
        try:
            from flwr.common import FitIns, Parameters
        except ImportError:
            pytest.skip("Flower not installed")

        # Create Flower-compatible inputs
        parameters = Parameters([np.array([1.0, 2.0])], "numpy")
        fit_ins = FitIns(
            parameters=parameters,
            config={
                "local_epochs": 1,
                "batch_size": 4,
                "learning_rate": 1e-4
            }
        )

        result = mock_client_with_model.fit(fit_ins)

        # Verify Flower API contract
        assert hasattr(result, 'parameters')
        assert hasattr(result, 'num_examples')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'status')
        assert result.status.code.value == 0  # OK

        # Verify metrics contain expected keys
        assert "loss" in result.metrics

    def test_evaluate_basic_workflow(self, mock_client_with_model):
        """Test evaluate method with basic Flower workflow."""
        try:
            from flwr.common import EvaluateIns, Parameters
        except ImportError:
            pytest.skip("Flower not installed")

        # Create Flower-compatible inputs
        parameters = Parameters([np.array([1.0, 2.0])], "numpy")
        evaluate_ins = EvaluateIns(
            parameters=parameters,
            config={}
        )

        result = mock_client_with_model.evaluate(evaluate_ins)

        # Verify Flower API contract
        assert hasattr(result, 'loss')
        assert hasattr(result, 'num_examples')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'status')
        assert result.status.code.value == 0  # OK

        # Verify loss is a number
        assert isinstance(result.loss, (int, float))

    def test_client_initialization_config(self, sample_client_config):
        """Test client initialization with configuration."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model loading disabled for test")

            client = SmolVLAClient(**sample_client_config)

            # Verify configuration is stored
            assert client.model_name == sample_client_config["model_name"]
            assert client.device == sample_client_config["device"]
            assert client.partition_id == sample_client_config["partition_id"]
            assert client.num_partitions == sample_client_config["num_partitions"]

    def test_simulate_training_step(self, sample_client_config):
        """Test training step simulation fallback."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**sample_client_config)

            loss = client._simulate_training_step()

            assert isinstance(loss, float)
            assert 0.1 <= loss <= 0.6

    def test_simulate_validation_step(self, sample_client_config):
        """Test validation step simulation fallback."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**sample_client_config)

            loss, correct = client._simulate_validation_step()

            assert isinstance(loss, float)
            assert isinstance(correct, int)
            assert 0.1 <= loss <= 0.4
            assert 0 <= correct <= 4