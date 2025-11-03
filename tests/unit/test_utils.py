"""Unit tests for utility functions in src/utils.py."""

import pytest
from unittest.mock import patch, mock_open
import numpy as np
import torch


class TestUtils:
    """Test utility functions."""

    def test_has_image_keys_with_image_keys(self):
        """Test has_image_keys detects image keys."""
        from src.core.utils import has_image_keys

        sample = {
            "observation": {
                "images": {
                    "camera1": torch.randn(3, 224, 224),
                    "camera2": torch.randn(3, 224, 224)
                }
            },
            "action": torch.randn(7)
        }

        assert has_image_keys(sample) is True

    def test_has_image_keys_without_image_keys(self):
        """Test has_image_keys returns False when no image keys."""
        from src.core.utils import has_image_keys

        sample = {
            "observation": {
                "state": torch.randn(10)
            },
            "action": torch.randn(7)
        }

        assert has_image_keys(sample) is False

    def test_has_image_keys_nested_structure(self):
        """Test has_image_keys with deeply nested structure."""
        from src.core.utils import has_image_keys

        sample = {
            "data": {
                "sensor": {
                    "camera": {
                        "rgb_image": torch.randn(3, 224, 224)
                    }
                }
            }
        }

        assert has_image_keys(sample) is True

    def test_compute_parameter_hash(self):
        """Test parameter hash computation."""
        from src.core.utils import compute_parameter_hash

        # Create test parameters
        params = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([5.0, 6.0, 7.0])
        ]

        hash1 = compute_parameter_hash(params)
        hash2 = compute_parameter_hash(params)

        # Same parameters should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex length

    def test_compute_parameter_hash_different_params(self):
        """Test parameter hash differs for different parameters."""
        from src.core.utils import compute_parameter_hash

        params1 = [np.array([1.0, 2.0])]
        params2 = [np.array([2.0, 1.0])]

        hash1 = compute_parameter_hash(params1)
        hash2 = compute_parameter_hash(params2)

        assert hash1 != hash2

    @patch('src.core.utils.logger')
    def test_validate_and_log_parameters_valid(self, mock_logger):
        """Test parameter validation with valid parameters."""
        from src.core.utils import validate_and_log_parameters

        # Create 506 test parameters (matching SmolVLA)
        params = [np.array([1.0, 2.0]) for _ in range(506)]

        hash_result = validate_and_log_parameters(params, "test_gate", 506)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
        mock_logger.info.assert_called()

    @patch('src.core.utils.logger')
    def test_validate_and_log_parameters_wrong_count(self, mock_logger):
        """Test parameter validation fails with wrong count."""
        from src.core.utils import validate_and_log_parameters

        params = [np.array([1.0, 2.0]) for _ in range(500)]  # Wrong count

        with pytest.raises(AssertionError, match="Parameter count mismatch"):
            validate_and_log_parameters(params, "test_gate", 506)

    def test_get_tool_config(self):
        """Test loading tool config from pyproject.toml."""
        from src.core.utils import get_tool_config

        mock_toml_content = """
[tool.test_tool]
key1 = "value1"
key2 = 42

[tool.other_tool]
key3 = "value3"
"""

        with patch('builtins.open', mock_open(read_data=mock_toml_content)):
            with patch('src.core.utils.tomllib.load') as mock_load:
                mock_load.return_value = {
                    "tool": {
                        "test_tool": {"key1": "value1", "key2": 42},
                        "other_tool": {"key3": "value3"}
                    }
                }

                config = get_tool_config("test_tool")
                assert config == {"key1": "value1", "key2": 42}

    def test_get_tool_config_missing_tool(self):
        """Test loading config for non-existent tool."""
        from src.core.utils import get_tool_config

        mock_toml_content = """
[tool.existing_tool]
key = "value"
"""

        with patch('builtins.open', mock_open(read_data=mock_toml_content)):
            with patch('src.core.utils.tomllib.load') as mock_load:
                mock_load.return_value = {
                    "tool": {
                        "existing_tool": {"key": "value"}
                    }
                }

                config = get_tool_config("missing_tool")
                assert config == {}

    def test_get_tool_config_missing_file(self):
        """Test handling missing pyproject.toml file."""
        from src.core.utils import get_tool_config

        with patch('builtins.open', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                get_tool_config("test_tool")