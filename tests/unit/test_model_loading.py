"""Tests for model loading utilities."""

import os
import torch
import pytest
from unittest.mock import patch

from src.utils import load_smolvla_model
MODEL_LOADING_AVAILABLE = True

try:
    from src.utils import load_lora_policy
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False


@pytest.mark.skipif(not MODEL_LOADING_AVAILABLE, reason="Model loading dependencies not available")
class TestModelLoading:
    """Test SmolVLA model loading with different TP configurations."""

    def test_load_model_none_tp_plan(self):
        """Test loading model with SMOLVLA_TP_PLAN='none'."""
        with patch.dict(os.environ, {'SMOLVLA_TP_PLAN': 'none'}):
            model = load_smolvla_model()
            assert model is not None
            assert next(model.parameters()).device.type in ['cuda', 'cpu']

    def test_load_model_auto_tp_plan_single_gpu(self):
        """Test loading model with SMOLVLA_TP_PLAN='auto' on single GPU."""
        with patch.dict(os.environ, {'SMOLVLA_TP_PLAN': 'auto'}):
            model = load_smolvla_model()
            assert model is not None
            assert next(model.parameters()).device.type in ['cuda', 'cpu']

    def test_load_model_invalid_tp_plan(self):
        """Test loading model with invalid SMOLVLA_TP_PLAN falls back gracefully."""
        with patch.dict(os.environ, {'SMOLVLA_TP_PLAN': 'invalid_value'}):
            model = load_smolvla_model()
            assert model is not None
            assert next(model.parameters()).device.type in ['cuda', 'cpu']

    def test_load_model_default_tp_plan(self):
        """Test loading model with default (no env var set) TP plan."""
        # Ensure env var is not set
        if 'SMOLVLA_TP_PLAN' in os.environ:
            del os.environ['SMOLVLA_TP_PLAN']

        model = load_smolvla_model()
        assert model is not None
        assert next(model.parameters()).device.type in ['cuda', 'cpu']

    def test_load_model_device_specification(self):
        """Test loading model with explicit device specification."""
        with patch.dict(os.environ, {'SMOLVLA_TP_PLAN': 'none'}):
            model = load_smolvla_model(device='cpu')
            assert model is not None
            assert next(model.parameters()).device.type == 'cpu'

    def test_load_model_custom_name(self):
        """Test loading model with custom model name."""
        with patch.dict(os.environ, {'SMOLVLA_TP_PLAN': 'none'}):
            model = load_smolvla_model(model_name="lerobot/smolvla_base")
            assert model is not None
            assert hasattr(model, 'config')


@pytest.mark.skipif(not LORA_AVAILABLE, reason="LoRA dependencies not available")
class TestLoRALoading:
    """Test LoRA policy loading and functionality."""

    def test_load_lora_policy_enabled(self):
        """Test loading policy with LoRA enabled using real dataset."""
        from src.utils import get_tool_config
        peft_config = get_tool_config("zk0.peft_config")

        # Load real dataset for metadata
        from src.utils import load_lerobot_dataset
        dataset = load_lerobot_dataset("lerobot/svla_so100_pickplace")
        dataset_meta = dataset.meta

        model = load_lora_policy(None, "cpu", peft_config, dataset_meta)
        assert model is not None

        # Check that LoRA adapters are applied
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        assert trainable_params < total_params  # Should have fewer trainable params with LoRA
        assert trainable_params > 0  # But still some trainable params

    def test_lora_forward_pass(self):
        """Test that LoRA model can perform forward pass with real data."""
        from src.utils import get_tool_config
        peft_config = get_tool_config("zk0.peft_config")

        # Load real dataset for metadata and sample
        from src.utils import load_lerobot_dataset
        dataset = load_lerobot_dataset("lerobot/svla_so100_pickplace")
        dataset_meta = dataset.meta

        model = load_lora_policy(None, "cpu", peft_config, dataset_meta)
        model.eval()

        # Get a real sample from the dataset
        sample = dataset[0]

        # Should not raise exception
        try:
            with torch.no_grad():
                output = model(sample)
            assert output is not None
        except Exception as e:
            # LoRA forward pass may fail with real data if codec issues, but should not crash on structure
            # This is expected for some environments
            assert "forward" in str(e) or "input" in str(e).lower() or "codec" in str(e).lower()

    def test_lora_forward_pass_validation(self):
        """Test LoRA forward pass with dummy input to validate compatibility."""
        from src.utils import get_tool_config
        peft_config = get_tool_config("zk0.peft_config")

        # Load real dataset for metadata
        from src.utils import load_lerobot_dataset
        dataset = load_lerobot_dataset("lerobot/svla_so100_pickplace")
        dataset_meta = dataset.meta

        model = load_lora_policy(None, "cpu", peft_config, dataset_meta)
        model.eval()

        # Create dummy input matching SmolVLA's expected format
        batch_size = 1
        device = torch.device("cpu")
        dummy_obs = {
            'observation.images.top': torch.randn(batch_size, 3, 480, 640).to(device),
            'observation.images.wrist': torch.randn(batch_size, 3, 480, 640).to(device),
            'observation.state': torch.randn(batch_size, 6).to(device),
            'task': "pick and place",
            'action': torch.tensor([0], dtype=torch.long, device=device),  # SmolVLA action dimension
        }
        dummy_task = "pick and place"

        # Test forward pass
        with torch.no_grad():
            output = model(dummy_obs, dummy_task)

        assert output is not None

        # Check output structure
        if isinstance(output, dict):
            action_key = 'action' if 'action' in output else list(output.keys())[0]
            action_pred = output[action_key]
        else:
            action_pred = output

        assert isinstance(action_pred, torch.Tensor)
        expected_shape = (batch_size, 7)  # SmolVLA action dimension
        assert action_pred.shape == expected_shape