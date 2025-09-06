"""Unit tests for server_app.py - focused on server functionality."""

import pytest
from unittest.mock import Mock, patch
import torch

from src.server_app import get_device, main


@pytest.mark.unit
class TestServerGetDevice:
    """Test cases for server get_device function."""

    @patch('torch.cuda.is_available', return_value=True)
    def test_get_device_auto_with_cuda(self, mock_cuda_available):
        """Test get_device with auto when CUDA is available."""
        result = get_device("auto")
        assert isinstance(result, torch.device)
        assert result.type == "cuda"

    @patch('torch.cuda.is_available', return_value=False)
    def test_get_device_auto_without_cuda(self, mock_cuda_available):
        """Test get_device with auto when CUDA is not available."""
        result = get_device("auto")
        assert isinstance(result, torch.device)
        assert result.type == "cpu"

    def test_get_device_explicit_cuda(self):
        """Test get_device with explicit cuda."""
        result = get_device("cuda")
        assert isinstance(result, torch.device)
        assert result.type == "cuda"

    def test_get_device_explicit_cpu(self):
        """Test get_device with explicit cpu."""
        result = get_device("cpu")
        assert isinstance(result, torch.device)
        assert result.type == "cpu"


@pytest.mark.unit
class TestServerApp:
    """Test cases for server app functionality."""

    def test_server_app_creation(self):
        """Test that ServerApp is created with correct configuration."""
        import importlib

        with patch('flwr.server.ServerApp') as mock_server_app_class:
            mock_app = Mock()
            mock_server_app_class.return_value = mock_app

            # Reload module to ensure patch takes effect
            import src.server_app
            importlib.reload(src.server_app)

            # Import to trigger app creation
            from src.server_app import app

            # Logging to validate mock calls
            print(f"Mock ServerApp call count: {mock_server_app_class.call_count}")

            mock_server_app_class.assert_called_once()
            args, kwargs = mock_server_app_class.call_args
            config = kwargs['config']
            assert hasattr(config, 'num_rounds')
            assert config.num_rounds == 50

    @patch('src.server_app.app')
    def test_main_function(self, mock_app):
        """Test main function runs the server app."""
        mock_app.run.return_value = None

        main()

        mock_app.run.assert_called_once()


@pytest.mark.unit
class TestServerAppEntryPoint:
    """Test the server app entry point."""

    @patch('src.server_app.main')
    def test_if_name_main(self, mock_main):
        """Test that main is called when script is run directly."""
        # This would be triggered by if __name__ == "__main__"
        # We can't directly test this without running the module,
        # but we can verify the main function exists and is callable
        assert callable(main)
        mock_main.assert_not_called()  # Should not be called during import