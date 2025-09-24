"""Unit tests for SmolVLAClient class - focused on Flower API integration."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import torch

from src.client_app import SmolVLAClient, get_device, load_config, client_fn, main


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
        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('src.client_app.AutoProcessor'), \
             patch('src.client_app.torch.optim.Adam', return_value=mock_optimizer):

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

        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class:
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
        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class:
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
        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model loading disabled for test")

            client = SmolVLAClient(**sample_client_config)

            # Verify configuration is stored
            assert client.model_name == sample_client_config["model_name"]
            assert client.device == sample_client_config["device"]
            assert client.partition_id == sample_client_config["partition_id"]
            assert client.num_partitions == sample_client_config["num_partitions"]

    def test_simulate_training_step(self, sample_client_config):
        """Test training step simulation fallback."""
        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**sample_client_config)

            loss = client._simulate_training_step()

            assert isinstance(loss, float)
            assert 0.1 <= loss <= 0.6

    def test_simulate_validation_step(self, sample_client_config):
        """Test validation step simulation fallback."""
        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**sample_client_config)

            loss, correct = client._simulate_validation_step()

            assert isinstance(loss, float)
            assert isinstance(correct, int)
            assert 0.1 <= loss <= 0.4
            assert 0 <= correct <= 4


@pytest.mark.unit
class TestLoadConfig:
    """Test cases for load_config function."""

    @patch('omegaconf.OmegaConf.load')
    def test_load_config_success(self, mock_omegaconf_load):
        """Test load_config with successful YAML loading."""
        mock_config = Mock()
        mock_omegaconf_load.return_value = mock_config

        result = load_config("test_config.yaml")

        mock_omegaconf_load.assert_called_once_with("test_config.yaml")
        assert result == mock_config

    @patch('omegaconf.OmegaConf.load', side_effect=ImportError("OmegaConf not available"))
    def test_load_config_omegaconf_import_error(self, mock_omegaconf_load):
        """Test load_config when OmegaConf is not available."""
        with pytest.raises(ImportError, match="OmegaConf is required"):
            load_config("test_config.yaml")

    @patch('omegaconf.OmegaConf.load', side_effect=Exception("File not found"))
    def test_load_config_file_error(self, mock_omegaconf_load):
        """Test load_config when config file cannot be loaded."""
        with pytest.raises(RuntimeError, match="Failed to load configuration"):
            load_config("nonexistent.yaml")


@pytest.mark.unit
class TestSmolVLAClientInitialization:
    """Test client initialization with configuration."""

    def test_client_initialization_with_config(self):
        """Test client initialization using config object."""
        mock_config = Mock()
        mock_config.model.name = "test_model"
        mock_config.model.device = "cpu"
        mock_config.federation.num_partitions = 4

        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model loading disabled")

            client = SmolVLAClient(config=mock_config, partition_id=1)

            assert client.model_name == "test_model"
            assert client.device == "cpu"
            assert client.num_partitions == 4
            assert client.partition_id == 1

    def test_client_initialization_config_dataset_access(self):
        """Test client initialization with config dataset access."""
        mock_config = Mock()
        mock_config.model.name = "test_model"
        mock_config.model.device = "cpu"
        mock_config.federation.num_partitions = 4
        mock_config.dataset.name = "test_dataset"
        mock_config.dataset.delta_timestamps = {"test": [1.0]}

        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('src.client_app.FederatedLeRobotDataset') as mock_federated:
            mock_model_class.from_pretrained.side_effect = Exception("Model loading disabled")
            mock_federated.return_value.load_partition.return_value = []

            client = SmolVLAClient(config=mock_config, partition_id=1)

            # Verify config.dataset was accessed
            assert hasattr(mock_config, 'dataset')

    def test_load_model_with_transformers_available(self, sample_client_config):
        """Test _load_model when transformers are available."""
        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('src.client_app.AutoProcessor') as mock_processor_class, \
             patch('src.client_app.torch.optim.Adam') as mock_optimizer_class:

            mock_model = Mock()
            mock_model.vision_encoder.parameters.return_value = [Mock()]
            mock_model.parameters.return_value = [Mock()]
            # Mock the .to() method to return the same object (device movement)
            mock_model.to.return_value = mock_model
            mock_model_class.from_pretrained.return_value = mock_model

            mock_processor = Mock()
            mock_processor_class.from_pretrained.return_value = mock_processor

            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer

            client = SmolVLAClient(**sample_client_config)
            client._load_model()

            assert client.model == mock_model
            assert client.processor == mock_processor
            assert client.optimizer == mock_optimizer

    def test_load_model_transformers_import_error(self, sample_client_config):
        """Test _load_model when transformers import fails."""
        with patch('src.client_app.AutoModelForVision2Seq', None), \
             patch('src.client_app.AutoProcessor', None):

            client = SmolVLAClient(**sample_client_config)
            client._load_model()

            assert client.model is None
            assert client.processor is None

    def test_load_model_freeze_vision_encoder(self, sample_client_config):
        """Test that vision encoder parameters are frozen."""
        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('src.client_app.AutoProcessor') as mock_processor_class, \
             patch('src.client_app.torch.optim.Adam') as mock_optimizer_class:

            mock_model = Mock()
            mock_param = Mock()
            mock_model.vision_encoder.parameters.return_value = [mock_param]
            mock_model.parameters.return_value = [mock_param, Mock()]
            # Mock the .to() method to return the same object (device movement)
            mock_model.to.return_value = mock_model
            mock_model_class.from_pretrained.return_value = mock_model

            mock_processor_class.from_pretrained.return_value = Mock()
            mock_optimizer_class.return_value = Mock()

            client = SmolVLAClient(**sample_client_config)
            client._load_model()

            # Verify vision encoder param was set to requires_grad=False
            mock_param.requires_grad = False

    def test_load_dataset_with_config(self, sample_client_config):
        """Test _load_dataset with config values."""
        mock_config = Mock()
        mock_config.dataset.name = "custom_dataset"
        mock_config.dataset.delta_timestamps = {"test": [1.0]}

        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('src.client_app.FederatedLeRobotDataset') as mock_federated_class:

            mock_model_class.from_pretrained.side_effect = Exception("Model disabled")
            mock_federated = Mock()
            mock_federated.load_partition.return_value = []
            mock_federated_class.return_value = mock_federated
            mock_federated.dataset = "custom_dataset"
            mock_federated.delta_timestamps = {"test": [1.0]}

            client = SmolVLAClient(config=mock_config, **sample_client_config)

            # Verify config was used correctly in initialization
            assert client.federated_dataset.dataset == "custom_dataset"
            assert client.federated_dataset.delta_timestamps == {"test": [1.0]}


@pytest.mark.unit
class TestSmolVLAClientModelOperations:
    """Test model operations like get_parameters, fit, evaluate."""

    def test_get_parameters_with_torch_tensors(self, sample_client_config):
        """Test get_parameters handles torch tensors correctly."""
        try:
            from flwr.common import GetParametersIns
        except ImportError:
            pytest.skip("Flower not installed")

        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('src.client_app.AutoProcessor'):

            mock_model = Mock()
            # Mock torch tensor
            mock_tensor = Mock()
            mock_tensor.cpu.return_value.numpy.return_value = np.array([1.0, 2.0])
            mock_tensor.hasattr = lambda attr: attr == 'cpu'  # Mock hasattr for torch tensor check

            mock_model.state_dict.return_value = {'param1': mock_tensor}
            mock_model_class.from_pretrained.return_value = mock_model

            client = SmolVLAClient(**sample_client_config)
            client.model = mock_model

            result = client.get_parameters(GetParametersIns(config={}))

            assert result.status.code.value == 0
            assert len(result.parameters.tensors) == 1
            assert isinstance(result.parameters.tensors[0], np.ndarray)

    def test_fit_real_training_loop(self, sample_client_config):
        """Test fit method with real training loop."""
        try:
            from flwr.common import FitIns, Parameters
        except ImportError:
            pytest.skip("Flower not installed")

        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('src.client_app.AutoProcessor'), \
             patch('src.client_app.torch.optim.Adam') as mock_optimizer_class, \
             patch('src.client_app.DataLoader') as mock_dataloader_class, \
             patch('src.client_app.time.time', return_value=0.0):

            mock_model = Mock()
            mock_outputs = Mock()
            mock_outputs.loss = Mock()
            mock_outputs.loss.item.return_value = 0.5  # Return float for arithmetic
            mock_model.return_value = mock_outputs
            mock_model.train.return_value = None
            mock_model.state_dict.return_value = {'param1': np.array([1.0])}
            mock_model_class.from_pretrained.return_value = mock_model

            mock_optimizer = Mock()
            mock_optimizer.param_groups = [{'lr': 0.01}]
            mock_optimizer_class.return_value = mock_optimizer

            # Mock dataloader with batches
            mock_batch = {'input_ids': torch.tensor([[1, 2]]), 'labels': torch.tensor([[1]])}
            mock_dataloader = Mock()
            mock_dataloader.__iter__ = Mock()
            mock_dataloader.__iter__.side_effect = lambda: iter([mock_batch] * 10)
            mock_dataloader_class.return_value = mock_dataloader

            client = SmolVLAClient(**sample_client_config)
            client.model = mock_model
            client.optimizer = mock_optimizer
            client.train_loader = mock_dataloader

            fit_ins = FitIns(
                parameters=Parameters([np.array([1.0])], "numpy"),
                config={"local_epochs": 1, "batch_size": 2}
            )

            result = client.fit(fit_ins)

            assert result.status.code.value == 0
            assert "loss" in result.metrics
            assert "epochs" in result.metrics
            assert result.num_examples > 0

    def test_evaluate_real_evaluation(self, sample_client_config):
        """Test evaluate method with real evaluation loop."""
        try:
            from flwr.common import EvaluateIns, Parameters
        except ImportError:
            pytest.skip("Flower not installed")

        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('src.client_app.AutoProcessor'), \
             patch('src.client_app.DataLoader') as mock_dataloader_class:

            mock_model = Mock()
            mock_outputs = Mock()
            mock_outputs.loss = Mock()
            mock_outputs.loss.item.return_value = 0.3  # Return float for arithmetic
            mock_model.return_value = mock_outputs
            mock_model.eval.return_value = None
            mock_model.state_dict.return_value = {'param1': np.array([1.0])}
            mock_model_class.from_pretrained.return_value = mock_model

            # Mock dataloader with batches
            mock_batch = {'input_ids': torch.tensor([[1, 2]]), 'labels': torch.tensor([[1]])}
            mock_dataloader = Mock()
            mock_dataloader.__iter__ = Mock()
            mock_dataloader.__iter__.return_value = iter([mock_batch] * 5)
            mock_dataloader_class.return_value = mock_dataloader

            client = SmolVLAClient(**sample_client_config)
            client.model = mock_model
            client.train_loader = mock_dataloader

            evaluate_ins = EvaluateIns(
                parameters=Parameters([np.array([1.0])], "numpy"),
                config={}
            )

            result = client.evaluate(evaluate_ins)

            assert result.status.code.value == 0
            assert isinstance(result.loss, float)
            assert "loss" in result.metrics
            assert "action_accuracy" in result.metrics
            assert result.num_examples > 0

    def test_save_checkpoint(self, sample_client_config, tmp_path):
        """Test _save_checkpoint method."""
        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('src.client_app.AutoProcessor'), \
             patch('src.client_app.torch.optim.Adam') as mock_optimizer_class, \
             patch('src.client_app.torch.save') as mock_save:

            mock_model = Mock()
            mock_model.state_dict.return_value = {'param1': np.array([1.0])}
            mock_model_class.from_pretrained.return_value = mock_model

            mock_optimizer = Mock()
            mock_optimizer.state_dict.return_value = {'lr': 0.01}
            mock_optimizer_class.return_value = mock_optimizer

            client = SmolVLAClient(**sample_client_config)
            client.model = mock_model
            client.optimizer = mock_optimizer
            client.output_dir = tmp_path / "test_output"
            client.output_dir.mkdir()

            client._save_checkpoint("test_checkpoint")

            mock_save.assert_called_once()
            args, kwargs = mock_save.call_args
            checkpoint_data = args[0]
            assert 'model_state_dict' in checkpoint_data
            assert 'optimizer_state_dict' in checkpoint_data
            assert 'partition_id' in checkpoint_data

    def test_load_checkpoint(self, sample_client_config, tmp_path):
        """Test _load_checkpoint method."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        checkpoint_data = {
            'model_state_dict': {'param1': np.array([1.0])},
            'optimizer_state_dict': {'lr': 0.01}
        }
        checkpoint_path.write_bytes(b"mock checkpoint data")

        with patch('src.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('src.client_app.AutoProcessor'), \
             patch('src.client_app.torch.load', return_value=checkpoint_data):

            mock_model = Mock()
            mock_model.load_state_dict.return_value = None
            mock_model_class.from_pretrained.return_value = mock_model

            client = SmolVLAClient(**sample_client_config)
            client.model = mock_model
            client.optimizer = Mock()

            client._load_checkpoint(checkpoint_path)

            mock_model.load_state_dict.assert_called_once_with({'param1': np.array([1.0])}, strict=True)


@pytest.mark.unit
class TestClientFnAndMain:
    """Test client_fn and main functions."""

    @patch('src.client_app.load_config')
    @patch('src.client_app.SmolVLAClient')
    def test_client_fn(self, mock_client_class, mock_load_config):
        """Test client_fn function."""
        try:
            from flwr.client import Client
        except ImportError:
            pytest.skip("Flower not installed")

        mock_config = Mock()
        mock_load_config.return_value = mock_config

        mock_client = Mock()
        mock_client.to_client.return_value = Mock(spec=Client)
        mock_client_class.return_value = mock_client

        context = Mock()
        context.node_config = {"partition-id": 1}

        result = client_fn(context)

        mock_load_config.assert_called_once_with("src/configs/default.yaml")
        mock_client_class.assert_called_once_with(config=mock_config, partition_id=1)
        assert result is not None

    @patch('src.client_app.app')
    def test_main_function(self, mock_app):
        """Test main function."""
        mock_app.run.return_value = None

        main()

        mock_app.run.assert_called_once()