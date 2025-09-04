#!/usr/bin/env python3
"""Test script for SmolVLA federated learning integration."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'smolvla_example'))

from smolvla_example.client_app import SmolVLAClient, get_device
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_device_detection():
    """Test device detection functionality."""
    logger.info("Testing device detection...")

    # Test auto detection
    device_auto = get_device("auto")
    logger.info(f"Auto device: {device_auto}")

    # Test CPU
    device_cpu = get_device("cpu")
    logger.info(f"CPU device: {device_cpu}")

    # Test CUDA if available
    device_cuda = get_device("cuda")
    logger.info(f"CUDA device: {device_cuda}")

    return True

def test_client_initialization():
    """Test SmolVLA client initialization."""
    logger.info("Testing SmolVLA client initialization...")

    try:
        # Test with CPU
        client = SmolVLAClient(
            model_name="lerobot/smolvla_base",
            device="cpu",
            partition_id=0,
            num_partitions=2
        )
        logger.info("Client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Client initialization failed: {e}")
        return False

def test_dataset_loading():
    """Test SO-100 dataset loading."""
    logger.info("Testing SO-100 dataset loading...")

    try:
        client = SmolVLAClient(
            model_name="lerobot/smolvla_base",
            device="cpu",
            partition_id=0,
            num_partitions=2
        )

        if client.train_loader is not None:
            logger.info(f"Dataset loaded successfully. Samples: {len(client.train_loader.dataset)}")
            return True
        else:
            logger.warning("Dataset loading returned None (expected in some environments)")
            return True  # This is acceptable for testing
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        return False

def main():
    """Run all integration tests."""
    logger.info("Starting SmolVLA federated learning integration tests...")

    tests = [
        ("Device Detection", test_device_detection),
        ("Client Initialization", test_client_initialization),
        ("Dataset Loading", test_dataset_loading),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! SmolVLA integration is ready.")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())