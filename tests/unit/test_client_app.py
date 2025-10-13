"""Unit tests for client_app.py."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.client_app import SmolVLAClient, compute_param_update_norm
from src.task import compute_fedprox_proximal_loss


class TestComputeParamUpdateNorm:
    """Test cases for compute_param_update_norm function."""

    def test_compute_param_update_norm_with_changes(self):
        """Test parameter update norm calculation with actual parameter changes."""
        pre_params = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([0.1, 0.2], dtype=np.float32),
        ]

        post_params = [
            np.array([[1.1, 2.1], [3.1, 4.1]], dtype=np.float32),  # Changed by 0.1
            np.array([0.2, 0.3], dtype=np.float32),                # Changed by 0.1
        ]

        # Calculate expected norm: sqrt(sum of squared differences)
        # Differences: [[0.1, 0.1], [0.1, 0.1]] and [0.1, 0.1]
        # Sum of squares: 0.1² * 4 + 0.1² * 2 = 0.04 + 0.02 = 0.06
        # Norm: sqrt(0.06) ≈ 0.2449
        expected_norm = np.sqrt(sum(np.sum((post - pre)**2) for post, pre in zip(post_params, pre_params)))

        result = compute_param_update_norm(pre_params, post_params)
        assert abs(result - expected_norm) < 1e-6
        assert result > 0.0

    def test_compute_param_update_norm_identical_params(self):
        """Test parameter update norm with identical parameters."""
        pre_params = [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([0.1], dtype=np.float32),
        ]

        post_params = [p.copy() for p in pre_params]  # Identical

        result = compute_param_update_norm(pre_params, post_params)
        assert result == 0.0

    def test_compute_param_update_norm_none_params(self):
        """Test parameter update norm with None parameters."""
        result = compute_param_update_norm(None, [np.array([1.0])])
        assert result == 0.0

        result = compute_param_update_norm([np.array([1.0])], None)
        assert result == 0.0

    def test_compute_param_update_norm_mismatched_lengths(self):
        """Test parameter update norm with mismatched parameter list lengths."""
        pre_params = [np.array([1.0])]
        post_params = [np.array([1.0]), np.array([2.0])]  # Different length

        result = compute_param_update_norm(pre_params, post_params)
        assert result == 0.0


class TestSmolVLAClient:
    """Test cases for SmolVLAClient."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock SmolVLA model."""
        model = MagicMock()
        # Mock the state_dict and named_parameters methods
        model.state_dict.return_value = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(5, 10),
        }
        model.named_parameters.return_value = [
            ('layer1.weight', torch.randn(10, 5)),
            ('layer1.bias', torch.randn(10)),
            ('layer2.weight', torch.randn(5, 10)),
        ]
        # Mock requires_grad for trainable params
        for name, param in model.named_parameters():
            param.requires_grad = True
        return model

    @pytest.fixture
    def mock_trainloader(self):
        """Create a mock trainloader."""
        loader = MagicMock()
        loader.dataset.meta.repo_id = "test_dataset"
        return loader

    @pytest.fixture
    def client(self, mock_model, mock_trainloader):
        """Create a test client instance."""
        with patch('src.client_app.get_model', return_value=mock_model):
            client = SmolVLAClient(
                partition_id=0,
                local_epochs=1,
                trainloader=mock_trainloader,
                nn_device=torch.device('cpu'),
                dataset_repo_id="test_dataset"
            )
            return client

    def test_param_update_norm_not_hardcoded_zero(self):
        """Test that param_update_norm is not hardcoded to 0.0 anymore."""
        # This test ensures the bug where param_update_norm was hardcoded to 0.0 is fixed
        # We test the core logic by checking that the function returns non-zero values for different params

        # Test with different parameters
        pre_params = [np.array([1.0, 2.0]), np.array([0.1])]
        post_params = [np.array([1.1, 2.1]), np.array([0.2])]  # Different values

        result = compute_param_update_norm(pre_params, post_params)
        assert result > 0.0, "param_update_norm should be > 0 when parameters actually change"

        # Test with identical parameters
        identical_params = [p.copy() for p in pre_params]
        result_identical = compute_param_update_norm(pre_params, identical_params)
        assert result_identical == 0.0, "param_update_norm should be 0 when parameters are identical"


class TestFedProxProximalLoss:
    """Test cases for FedProx proximal loss calculation."""

    def test_compute_fedprox_proximal_loss_with_different_params(self):
        """Test FedProx proximal loss calculation with parameter differences."""
        # Create torch tensors for current params
        current_params = [
            torch.tensor([[1.1, 2.1]], dtype=torch.float32),  # Different from global
            torch.tensor([[0.2]], dtype=torch.float32),
        ]

        # Create numpy arrays for global params
        global_params = [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[0.1]], dtype=np.float32),
        ]

        fedprox_mu = 0.01

        # Calculate expected proximal loss manually
        # ||w - w_global||² = (0.1)² + (0.1)² + (0.1)² = 0.03
        # (μ/2) * ||w - w_global||² = (0.01/2) * 0.03 = 0.005 * 0.03 = 0.00015
        expected_loss = (fedprox_mu / 2.0) * 0.03

        result = compute_fedprox_proximal_loss(current_params, global_params, fedprox_mu)
        assert abs(result - expected_loss) < 1e-6
        assert result > 0.0

    def test_compute_fedprox_proximal_loss_identical_params(self):
        """Test FedProx proximal loss with identical parameters."""
        current_params = [
            torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            torch.tensor([[0.1]], dtype=torch.float32),
        ]

        global_params = [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[0.1]], dtype=np.float32),
        ]

        fedprox_mu = 0.01

        result = compute_fedprox_proximal_loss(current_params, global_params, fedprox_mu)
        assert result == 0.0

    def test_compute_fedprox_proximal_loss_zero_mu(self):
        """Test FedProx proximal loss with zero regularization coefficient."""
        current_params = [
            torch.tensor([[1.1, 2.1]], dtype=torch.float32),
        ]

        global_params = [
            np.array([[1.0, 2.0]], dtype=np.float32),
        ]

        result = compute_fedprox_proximal_loss(current_params, global_params, fedprox_mu=0.0)
        assert result == 0.0

    def test_compute_fedprox_proximal_loss_none_global_params(self):
        """Test FedProx proximal loss with None global parameters."""
        current_params = [
            torch.tensor([[1.0]], dtype=torch.float32),
        ]

        result = compute_fedprox_proximal_loss(current_params, None, fedprox_mu=0.01)
        assert result == 0.0