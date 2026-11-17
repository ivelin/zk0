"""Regression: LeRobot's hub helpers need huggingface_hub 1.33+ APIs."""

from importlib.metadata import version

from packaging.version import Version


def test_huggingface_hub_exposes_download_bucket_files_for_lerobot():
    """Fail if zk0's resolver again downgrades huggingface_hub below LeRobot."""
    from huggingface_hub import download_bucket_files

    hub_version = Version(version("huggingface_hub"))
    assert hub_version >= Version("1.33.0")
    assert hub_version < Version("3.0.0")
    assert callable(download_bucket_files)

    from src.client_app import SmolVLAClient

    assert SmolVLAClient is not None
