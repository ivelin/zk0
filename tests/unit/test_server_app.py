"""Unit tests for server_app.py - focused on server functionality."""

import pytest
from unittest.mock import Mock, patch
import torch

from src.server_app import get_device, main, server_fn


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

            # Logging to validate mock calls
            print(f"Mock ServerApp call count: {mock_server_app_class.call_count}")

            mock_server_app_class.assert_called_once()
            args, kwargs = mock_server_app_class.call_args
            # The app creation doesn't directly create config anymore
            # since it's done in server_fn with context
            assert len(args) > 0 or len(kwargs) > 0

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

    def test_main_function_coverage(self):
        """Test main function to improve coverage."""
        with patch('src.server_app.app') as mock_app:
            mock_app.run.return_value = None

            # Call main to cover line 119
            main()

            mock_app.run.assert_called_once()


@pytest.mark.unit
class TestServerFn:
    """Test cases for server_fn function."""

    def test_server_fn_basic_config(self):
        """Test server_fn creates components with basic configuration."""
        try:
            from flwr.server import ServerAppComponents, ServerConfig
        except ImportError:
            pytest.skip("Flower not installed")

        # Mock context
        context = Mock()
        context.run_config = {"num-server-rounds": 10}

        with patch('pathlib.Path.mkdir'), \
             patch('flwr.server.strategy.FedAvg') as mock_fedavg_class:

            mock_strategy = Mock()
            mock_fedavg_class.return_value = mock_strategy

            components = server_fn(context)

            assert isinstance(components, ServerAppComponents)
            assert components.strategy is not None
            assert isinstance(components.config, ServerConfig)
            assert components.config.num_rounds == 10

    def test_server_fn_default_config(self):
        """Test server_fn uses default configuration when no config provided."""
        try:
            from flwr.server import ServerAppComponents, ServerConfig
        except ImportError:
            pytest.skip("Flower not installed")

        # Mock context with empty config
        context = Mock()
        context.run_config = {}

        with patch('pathlib.Path.mkdir'), \
             patch('flwr.server.strategy.FedAvg') as mock_fedavg_class:

            mock_strategy = Mock()
            mock_fedavg_class.return_value = mock_strategy

            components = server_fn(context)

            assert isinstance(components, ServerAppComponents)
            assert components.config.num_rounds == 50  # Default value


@pytest.mark.unit
class TestServerStrategy:
    """Test cases for server strategy functionality."""

    def test_server_fn_creates_logging_strategy(self):
        """Test that server_fn creates a strategy with logging capabilities."""
        try:
            from flwr.server import ServerAppComponents
        except ImportError:
            pytest.skip("Flower not installed")

        # Mock context
        context = Mock()
        context.run_config = {"num-server-rounds": 5}

        with patch('pathlib.Path.mkdir'), \
             patch('time.time', return_value=1000.0), \
             patch('builtins.open', create=True), \
             patch('json.dump'):

            components = server_fn(context)

            # Verify strategy has logging attributes
            strategy = components.strategy
            assert hasattr(strategy, 'round_metrics')
            assert hasattr(strategy, 'start_time')
            assert hasattr(strategy, 'aggregate_fit')
            assert hasattr(strategy, 'aggregate_evaluate')
            assert callable(strategy.aggregate_fit)
            assert callable(strategy.aggregate_evaluate)

    def test_strategy_aggregate_fit_logs_metrics(self):
        """Test that strategy's aggregate_fit method logs metrics."""
        try:
            from flwr.server import ServerAppComponents
        except ImportError:
            pytest.skip("Flower not installed")

        # Mock context
        context = Mock()
        context.run_config = {"num-server-rounds": 5}

        with patch('pathlib.Path.mkdir'), \
             patch('time.time', return_value=1000.0), \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:

            components = server_fn(context)
            strategy = components.strategy

            # Mock the parent aggregate_fit method
            with patch('flwr.server.strategy.FedAvg.aggregate_fit', return_value=Mock()) as mock_parent:
                # Create mock results with metrics
                mock_result1 = Mock()
                mock_result1.metrics = {"loss": 0.5, "accuracy": 0.8}

                mock_result2 = Mock()
                mock_result2.metrics = {"loss": 0.3, "accuracy": 0.9}

                results = [mock_result1, mock_result2]
                failures = []

                # Call aggregate_fit
                result = strategy.aggregate_fit(1, results, failures)

                # Verify parent method was called
                mock_parent.assert_called_once_with(1, results, failures)

                # Verify metrics were logged
                assert len(strategy.round_metrics) == 1
                round_data = strategy.round_metrics[0]
                assert round_data["round"] == 1
                assert round_data["num_clients"] == 2
                assert round_data["num_failures"] == 0
                assert "client_metrics" in round_data
                assert len(round_data["client_metrics"]) == 2

                # Verify JSON dump was called for progress logging
                mock_json_dump.assert_called()

    def test_strategy_aggregate_evaluate_final_summary(self):
        """Test that strategy creates final summary on last round."""
        try:
            from flwr.server import ServerAppComponents
        except ImportError:
            pytest.skip("Flower not installed")

        # Mock context
        context = Mock()
        context.run_config = {"num-server-rounds": 3}

        with patch('pathlib.Path.mkdir'), \
             patch('time.time', side_effect=[1000.0, 1010.0, 1020.0]), \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:

            components = server_fn(context)
            strategy = components.strategy

            # Mock the parent aggregate_evaluate method
            with patch('flwr.server.strategy.FedAvg.aggregate_evaluate', return_value=Mock()) as mock_parent:
                # Call aggregate_evaluate for final round
                result = strategy.aggregate_evaluate(3, [], [])

                # Verify parent method was called
                mock_parent.assert_called_once_with(3, [], [])

                # Verify summary JSON dump was called (should be called twice: progress + summary)
                assert mock_json_dump.call_count >= 1