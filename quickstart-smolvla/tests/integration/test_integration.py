"""Integration tests for SmolVLA federated learning - focused on Flower API integration."""

import pytest
from unittest.mock import patch, Mock
import numpy as np

from smolvla_example.client_app import SmolVLAClient, get_device


@pytest.mark.integration
class TestDeviceDetection:
    """Test device detection functionality."""

    @patch('torch.cuda.is_available', return_value=True)
    def test_device_detection_auto_with_cuda(self, mock_cuda_available):
        """Test device detection with auto when CUDA is available."""
        device = get_device("auto")
        assert device == "cuda"

    @patch('torch.cuda.is_available', return_value=False)
    def test_device_detection_auto_without_cuda(self, mock_cuda_available):
        """Test device detection with auto when CUDA is not available."""
        device = get_device("auto")
        assert device == "cpu"

    def test_device_detection_cpu(self):
        """Test device detection with explicit CPU."""
        device = get_device("cpu")
        assert device == "cpu"

    def test_device_detection_cuda(self):
        """Test device detection with explicit CUDA."""
        device = get_device("cuda")
        assert device == "cuda"


@pytest.mark.integration
class TestFlowerAPIIntegration:
    """Test Flower API integration for SmolVLAClient."""

    @pytest.fixture
    def flower_client(self, client_config, mock_model, mock_optimizer):
        """Create a client configured for Flower API testing."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('smolvla_example.client_app.AutoProcessor'), \
             patch('smolvla_example.client_app.torch.optim.Adam', return_value=mock_optimizer):

            mock_model_class.from_pretrained.return_value = mock_model

            client = SmolVLAClient(**client_config)
            client.model = mock_model
            client.optimizer = mock_optimizer
            return client

    def test_get_parameters_flower_contract(self, flower_client):
        """Test get_parameters conforms to Flower API contract."""
        try:
            from flwr.common import GetParametersIns
        except ImportError:
            pytest.skip("Flower not installed")

        result = flower_client.get_parameters(GetParametersIns(config={}))

        # Verify Flower API contract compliance
        assert hasattr(result, 'parameters')
        assert hasattr(result.parameters, 'tensors')
        assert isinstance(result.parameters.tensors, list)
        assert hasattr(result, 'status')
        assert result.status.code.value == 0  # OK

        # Verify all tensors are numpy arrays (Flower requirement)
        for tensor in result.parameters.tensors:
            assert isinstance(tensor, np.ndarray)

    def test_set_parameters_flower_contract(self, flower_client):
        """Test set_parameters conforms to Flower API contract."""
        # Flower sends parameters as list of numpy arrays
        parameters = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0])
        ]

        # Should complete without error
        flower_client.set_parameters(parameters)

        # Verify model received the parameters
        flower_client.model.load_state_dict.assert_called_once()

    def test_fit_flower_contract(self, flower_client):
        """Test fit method conforms to Flower API contract."""
        try:
            from flwr.common import FitIns, Parameters
        except ImportError:
            pytest.skip("Flower not installed")

        # Create Flower-compatible FitIns
        parameters = Parameters([np.array([1.0, 2.0])], "numpy")
        fit_ins = FitIns(
            parameters=parameters,
            config={
                "local_epochs": 1,
                "batch_size": 4,
                "learning_rate": 1e-4
            }
        )

        result = flower_client.fit(fit_ins)

        # Verify Flower API contract compliance
        assert hasattr(result, 'parameters')
        assert hasattr(result, 'num_examples')
        assert isinstance(result.num_examples, int)
        assert hasattr(result, 'metrics')
        assert isinstance(result.metrics, dict)
        assert hasattr(result, 'status')
        assert result.status.code.value == 0  # OK

    def test_evaluate_flower_contract(self, flower_client):
        """Test evaluate method conforms to Flower API contract."""
        try:
            from flwr.common import EvaluateIns, Parameters
        except ImportError:
            pytest.skip("Flower not installed")

        # Create Flower-compatible EvaluateIns
        parameters = Parameters([np.array([1.0, 2.0])], "numpy")
        evaluate_ins = EvaluateIns(
            parameters=parameters,
            config={}
        )

        result = flower_client.evaluate(evaluate_ins)

        # Verify Flower API contract compliance
        assert hasattr(result, 'loss')
        assert isinstance(result.loss, (int, float))
        assert hasattr(result, 'num_examples')
        assert isinstance(result.num_examples, int)
        assert hasattr(result, 'metrics')
        assert isinstance(result.metrics, dict)
        assert hasattr(result, 'status')
        assert result.status.code.value == 0  # OK

    def test_federated_workflow_end_to_end(self, flower_client):
        """Test complete federated learning workflow."""
        try:
            from flwr.common import GetParametersIns, FitIns, EvaluateIns, Parameters
        except ImportError:
            pytest.skip("Flower not installed")

        # Step 1: Get initial parameters
        get_params_result = flower_client.get_parameters(GetParametersIns(config={}))
        assert get_params_result.status.code.value == 0

        # Step 2: Server sends parameters for training
        initial_params = get_params_result.parameters

        # Step 3: Client receives parameters and trains
        fit_ins = FitIns(
            parameters=initial_params,
            config={"local_epochs": 1, "batch_size": 4}
        )
        fit_result = flower_client.fit(fit_ins)
        assert fit_result.status.code.value == 0

        # Step 4: Client sends updated parameters back
        updated_params = fit_result.parameters

        # Step 5: Server sends parameters for evaluation
        evaluate_ins = EvaluateIns(
            parameters=updated_params,
            config={}
        )
        evaluate_result = flower_client.evaluate(evaluate_ins)
        assert evaluate_result.status.code.value == 0

        # Verify the workflow completed successfully
        assert evaluate_result.loss >= 0
        assert evaluate_result.num_examples > 0

    def test_error_handling_graceful_degradation(self, client_config):
        """Test that client handles errors gracefully without breaking Flower workflow."""
        try:
            from flwr.common import GetParametersIns, FitIns, EvaluateIns, Parameters
        except ImportError:
            pytest.skip("Flower not installed")

        # Create client with model loading failure
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**client_config)

            # All Flower API methods should still work (return appropriate defaults)
            get_result = client.get_parameters(GetParametersIns(config={}))
            assert get_result.status.code.value == 0

            fit_result = client.fit(FitIns(
                parameters=Parameters([], "numpy"),
                config={}
            ))
            assert fit_result.status.code.value == 0

            eval_result = client.evaluate(EvaluateIns(
                parameters=Parameters([], "numpy"),
                config={}
            ))
            assert eval_result.status.code.value == 0

    def test_configuration_persistence(self, client_config):
        """Test that client configuration is properly maintained."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model disabled")

            client = SmolVLAClient(**client_config)

            # Verify configuration is stored correctly
            assert client.model_name == client_config["model_name"]
            assert client.device == client_config["device"]
            assert client.partition_id == client_config["partition_id"]
            assert client.num_partitions == client_config["num_partitions"]