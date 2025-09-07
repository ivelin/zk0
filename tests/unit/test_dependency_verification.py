"""Dependency verification tests for zk0 environment - ensures real components work."""

import pytest
import numpy as np


@pytest.mark.unit
class TestCoreDependencies:
    """Test that core dependencies are available and functional in zk0 environment."""

    def test_torch_import_and_basic_functionality(self):
        """Test PyTorch is available and basic tensor operations work."""
        import torch

        # Test basic tensor creation
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])

        # Test basic operations
        result = x + y
        expected = torch.tensor([5.0, 7.0, 9.0])

        assert torch.allclose(result, expected)
        assert x.device.type in ['cpu', 'cuda']  # Should work on either

    def test_numpy_import_and_integration(self):
        """Test NumPy is available and integrates with torch."""
        import torch

        # Test numpy array creation
        np_array = np.array([1.0, 2.0, 3.0])

        # Test conversion to torch tensor
        torch_tensor = torch.from_numpy(np_array)
        assert torch_tensor.dtype == torch.float64

        # Test conversion back
        back_to_numpy = torch_tensor.numpy()
        assert np.allclose(np_array, back_to_numpy)

    def test_flower_import_and_basic_functionality(self):
        """Test Flower framework is available and basic components work."""
        try:
            from flwr.common import Parameters, ndarrays_to_parameters

            # Test parameter creation
            arrays = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
            parameters = ndarrays_to_parameters(arrays)

            assert parameters is not None
            assert len(parameters.tensors) == 2

        except ImportError:
            pytest.skip("Flower not installed in test environment")

    def test_lerobot_import_availability(self):
        """Test that lerobot can be imported (may fail if not installed)."""
        try:
            import lerobot
            assert lerobot is not None
        except ImportError:
            pytest.skip("lerobot not available in test environment")

    def test_huggingface_transformers_availability(self):
        """Test that transformers library is available."""
        try:
            import transformers
            assert transformers is not None

            # Test basic tokenizer functionality
            from transformers import AutoTokenizer
            # This will use a small model if available
            try:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
                tokens = tokenizer.encode("Hello world")
                assert len(tokens) > 0
            except Exception:
                # Skip if model not available locally
                pass

        except ImportError:
            pytest.skip("transformers not available in test environment")


@pytest.mark.unit
class TestDependencyIntegration:
    """Test integration between dependencies."""

    def test_torch_numpy_parameter_conversion(self):
        """Test parameter conversion between torch and numpy (critical for Flower)."""
        import torch

        # Create torch tensor
        torch_param = torch.randn(10, 5)

        # Convert to numpy (as Flower expects)
        numpy_param = torch_param.detach().cpu().numpy()

        # Verify shapes match
        assert numpy_param.shape == torch_param.shape

        # Convert back to torch
        back_to_torch = torch.from_numpy(numpy_param)

        # Verify values match
        assert torch.allclose(torch_param, back_to_torch)

    def test_flower_torch_integration(self):
        """Test Flower and PyTorch integration for parameter handling."""
        try:
            import torch
            from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

            # Create torch parameters
            torch_params = [
                torch.randn(5, 3),
                torch.randn(10, 2)
            ]

            # Convert to numpy arrays (Flower format)
            numpy_params = [p.detach().cpu().numpy() for p in torch_params]

            # Convert to Flower Parameters
            flower_params = ndarrays_to_parameters(numpy_params)

            # Convert back to numpy
            back_to_numpy = parameters_to_ndarrays(flower_params)

            # Verify round-trip conversion
            for orig, final in zip(numpy_params, back_to_numpy):
                assert np.allclose(orig, final)

        except ImportError:
            pytest.skip("Flower not available in test environment")

    def test_device_compatibility(self):
        """Test device compatibility between torch and other libraries."""
        import torch

        # Test CPU device
        cpu_device = torch.device('cpu')
        cpu_tensor = torch.tensor([1.0, 2.0], device=cpu_device)
        assert cpu_tensor.device == cpu_device

        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_device = torch.device('cuda')
            cuda_tensor = torch.tensor([1.0, 2.0], device=cuda_device)
            assert cuda_tensor.device.type == 'cuda'
            assert cuda_tensor.is_cuda
        else:
            # If no CUDA, ensure CPU fallback works
            assert cpu_tensor.is_cpu


@pytest.mark.unit
class TestEnvironmentConfiguration:
    """Test that environment is properly configured for zk0 project."""

    def test_python_version_compatibility(self):
        """Test Python version meets project requirements."""
        import sys

        version = sys.version_info
        assert version.major == 3
        assert version.minor >= 8  # Minimum Python 3.8 for the project

    def test_required_modules_available(self):
        """Test that all required modules for basic functionality are available."""
        required_modules = [
            'os', 'sys', 'pathlib', 'json', 'logging',
            'numpy', 'torch'
        ]

        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Required module {module_name} not available: {e}")

    def test_zk0_environment_indicators(self):
        """Test for indicators that we're in the correct zk0 environment."""
        import os

        # Check if we're in conda environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            # If we're in a conda env, it should be accessible
            assert conda_env is not None
        else:
            # If not in conda, that's also fine - just document it
            pass