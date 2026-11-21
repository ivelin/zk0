"""Study 001 packaging step (D-11/G-8): write the serialized D-9
pooled preprocessor into a finished FL checkpoint directory,
replacing the base-stats preprocessor files save_model_checkpoint
copied there, then verify: (a) the packaged dir's preprocessor
hashes (driver's preprocessor_stats_hash) to the D-9 value
a0455b5b...; (b) the dir loads via SmolVLAPolicy.from_pretrained +
make_pre_post_processors (the eval stage's own construction).

Usage: package_final_checkpoint.py <checkpoint_dir>
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-20261119/run/scripts")

EXPECTED = "a0455b5ba7d75b9c3d743754828020ca751f29fe93447a106207553b89d7a39d"
POOLED = Path(
    "/home/openclaw/muse-work/zk0-study001-fl-20261121/study_assets/pooled_preprocessor"
)

ckpt = Path(sys.argv[1])
assert ckpt.is_dir(), ckpt
assert (ckpt / "model.safetensors").is_file(), "not a checkpoint dir"

import shutil

copied = []
for src in sorted(POOLED.iterdir()):
    if src.name.startswith("policy_preprocessor"):
        shutil.copy2(src, ckpt / src.name)
        copied.append(src.name)
print("copied pooled preprocessor files:", copied)
assert "policy_preprocessor.json" in copied

import fl_driver
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained(str(ckpt))
pre, _post = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=str(ckpt)
)
h = fl_driver.preprocessor_stats_hash(pre)
print("packaged preprocessor stats hash:", h)
assert h == EXPECTED, f"packaged stats hash mismatch: {h}"
print("PACKAGING PASS:", ckpt)
