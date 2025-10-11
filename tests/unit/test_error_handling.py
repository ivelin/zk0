"""Unit tests for error handling scenarios in SmolVLA - focused on Flower API robustness."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.client_app import SmolVLAClient


@pytest.mark.unit
class TestFitEvaluateExceptionHandling:
    """Test exception handling in fit and evaluate methods."""

    @pytest.fixture
    def client_config(self):
        """Default client configuration for tests."""
        from unittest.mock import MagicMock
        mock_trainloader = MagicMock()
        mock_trainloader.dataset.meta.repo_id = "test/repo"

        return {
            "partition_id": 0,
            "local_epochs": 1,
            "trainloader": mock_trainloader,
            "nn_device": "cpu",
            "dataset_repo_id": "test/repo"
        }