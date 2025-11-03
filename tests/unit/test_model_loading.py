"""Tests for model loading utilities."""

import os
import pytest
from unittest.mock import patch

try:
    from src.core.utils import load_smolvla_model
    MODEL_LOADING_AVAILABLE = True
except ImportError:
    MODEL_LOADING_AVAILABLE = False


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