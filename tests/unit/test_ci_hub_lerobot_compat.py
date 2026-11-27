"""Regression for CI image vs zk0's transformers pin.

On huggingface/lerobot-cpu (git main, hub 1.x + transformers 5.x), installing
zk0's transformers>=4.50.3,<4.52.0 pin drops huggingface_hub to 0.36.x and
collection then fails with:

    ImportError: cannot import name sync_bucket from huggingface_hub

See GitHub Actions run 45919343321 at da557a7.
"""

from importlib.metadata import version


def test_lerobot_imports_with_installed_huggingface_hub():
    import huggingface_hub
    import lerobot

    assert huggingface_hub.__version__
    assert lerobot.__version__

    # Intentional CI tripwire that changes when zk0 lifts its transformers pin.
    hub_version = version("huggingface_hub")
    assert hub_version.split(".", 1)[0] == "0", hub_version
