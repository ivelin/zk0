"""Staged sim assets (spec §1.3).

Assets are pulled from the Hugging Face collection at pinned versions
with content hashes; sealed assets are reused on hash match without
re-validation (SSOT decision 46). In this dev build slice the staging
source is the certified local copies (see dev/bootstrap.sh), staged
into the node's asset cache and md5-verified — the eval record names
the staged file's md5 either way, so a result traces to the exact
asset bytes that produced it.
"""
import hashlib
import os
from pathlib import Path

from ..config import zk0_home

VARIANTS = ("realistic", "stock")


def assets_dir() -> Path:
    return Path(os.environ.get("ZK0_NODE_ASSETS", zk0_home() / "assets"))


def model_dir(variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"unknown arm model variant {variant!r} ({VARIANTS})")
    return assets_dir() / f"model-{variant}"


def md5_of(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def arm_model_info(variant: str) -> dict:
    """Identity of the staged arm model, from the staged bytes."""
    d = model_dir(variant)
    info = {"variant": variant, "staged_dir": str(d), "present": d.exists()}
    if variant == "realistic":
        info.update({
            "name": "so101-realistic",
            "version": "0.2.0",
            "source": "HF zk0bot/so101-realistic (model files byte-identical "
                      "to v0.1.0); staged from the certified local copy",
            "canonical_file": "so101_realistic.xml",
        })
        canon = d / "so101_realistic.xml"
        if canon.exists():
            info["canonical_md5"] = md5_of(canon)
    else:
        info.update({
            "name": "robotstudio_so101 (stock)",
            "version": "MuJoCo Menagerie HEAD (certified multitask config)",
            "source": "certified multitask work dir (control runs only)",
            "canonical_file": "so101.xml",
        })
        canon = d / "so101.xml"
        if canon.exists():
            info["canonical_md5"] = md5_of(canon)
    live = d / "so101.xml"
    if live.exists():
        info["live_include_md5"] = md5_of(live)
    return info
