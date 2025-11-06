"""Unit tests for logger configuration functions in src/logger.py."""

from unittest.mock import patch, mock_open


class TestLoggerConfig:
    """Test logger configuration functions."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.common.utils.get_tool_config')
    def test_get_config_with_valid_config(self, mock_get_tool_config, mock_file):
        """Test get_config loads from pyproject.toml successfully."""
        from src.logger import get_config

        mock_get_tool_config.return_value = {
            "level": "INFO",
            "enable_grpc_logging": True,
            "log_format": "json"
        }

        config = get_config()

        assert config["level"] == "INFO"
        assert config["enable_grpc_logging"] is True
        assert config["log_format"] == "json"

    @patch('src.common.utils.get_tool_config', side_effect=Exception("Config loading failed"))
    def test_get_config_fallback_on_error(self, mock_get_tool_config):
        """Test get_config returns defaults when config loading fails."""
        from src.logger import get_config

        config = get_config()

        assert config["level"] == "DEBUG"
        assert config["flwr_log_level"] == "INFO"
        assert config["lerobot_log_level"] == "INFO"
        assert config["enable_grpc_logging"] is False
        assert config["log_format"] == "detailed"

    def test_get_formats_json(self):
        """Test get_formats returns JSON format when configured."""
        from src.logger import get_formats

        config = {"log_format": "json"}
        console_format, file_format = get_formats(config)

        assert "PID:{extra[process_id]}" in console_format
        assert "PID:{extra[process_id]}" in file_format

    def test_get_formats_detailed(self):
        """Test get_formats returns detailed format by default."""
        from src.logger import get_formats

        config = {"log_format": "detailed"}
        console_format, file_format = get_formats(config)

        assert "<green>" in console_format
        assert "<level>" in console_format
        assert "PID:{extra[process_id]}" in file_format

    def test_get_rotation_mb_mb(self):
        """Test get_rotation_mb parses MB values."""
        from src.logger import get_rotation_mb

        config = {"file_rotation": "100 MB"}
        result = get_rotation_mb(config)

        assert result == 100

    def test_get_rotation_mb_gb(self):
        """Test get_rotation_mb parses GB values."""
        from src.logger import get_rotation_mb

        config = {"file_rotation": "2 GB"}
        result = get_rotation_mb(config)

        assert result == 2048  # 2 * 1024

    def test_get_rotation_mb_default(self):
        """Test get_rotation_mb returns default for invalid format."""
        from src.logger import get_rotation_mb

        config = {"file_rotation": "invalid"}
        result = get_rotation_mb(config)

        assert result == 500

    def test_get_rotation_mb_missing(self):
        """Test get_rotation_mb returns default when missing."""
        from src.logger import get_rotation_mb

        config = {}
        result = get_rotation_mb(config)

        assert result == 500