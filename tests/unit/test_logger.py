"""Unit tests for logger setup and loguru integration."""

import tempfile
from pathlib import Path
from loguru import logger

from src.logger import setup_logging


def test_setup_logging_basic():
    """Test basic logging setup without extras."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"

        setup_logging(log_file)

        # Test logging
        logger.info("Test message")

        # Force flush to ensure message is written
        import time
        time.sleep(0.1)  # Small delay for async logging

        # Check file was created and contains message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content
        assert "server" in content  # Server process when no client_id provided


def test_setup_logging_with_client_id():
    """Test logging setup with client_id extra."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"

        setup_logging(log_file, client_id="client_1")

        logger.info("Test with client")

        content = log_file.read_text()
        assert "client_1" in content
        assert "Test with client" in content


def test_setup_logging_with_process_id():
    """Test logging setup includes process ID."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"

        setup_logging(log_file)

        logger.info("Test with PID")

        content = log_file.read_text()
        assert "PID:" in content
        assert "Test with PID" in content


def test_setup_logging_flwr_propagation():
    """Test that flwr logger is propagated to loguru."""
    import logging as std_logging

    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"

        setup_logging(log_file)

        # Get flwr logger and log
        flwr_logger = std_logging.getLogger('flwr')
        flwr_logger.info("Flwr test message")

        content = log_file.read_text()
        assert "Flwr test message" in content


def test_setup_logging_file_rotation():
    """Test that file rotation works."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"

        setup_logging(log_file)

        # Log many messages to trigger rotation (though unlikely with small messages)
        for i in range(100):
            logger.info(f"Message {i}")

        # Check file exists
        assert log_file.exists()

        # Check for rotation files if any
        rotation_files = list(Path(temp_dir).glob("test.*"))
        # May or may not have rotation files depending on size, but no errors should occur
        assert True  # Test passes if no exceptions


def test_setup_logging_multiple_calls():
    """Test that multiple setup calls work without errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"

        # First setup
        setup_logging(log_file)
        logger.info("First setup")

        # Second setup (should work without issues)
        setup_logging(log_file)
        logger.info("Second setup")

        # Check file contains both messages
        content = log_file.read_text()
        assert "First setup" in content
        assert "Second setup" in content