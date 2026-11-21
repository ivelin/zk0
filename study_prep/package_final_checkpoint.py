"""Study 001 packaging step (D-11/G-8): write the serialized D-9
pooled preprocessor into a finished FL checkpoint directory,
replacing the base-stats preprocessor files save_model_checkpoint
copied there, plus the tokenizer artifact and the postprocessor
unnormalizer state the substrate does not write. Then verify:
(a) the dir loads via SmolVLAPolicy.from_pretrained +
make_pre_post_processors (the eval stage's own construction);
(b) the packaged preprocessor's stats equal the pooled D-9 stats
at F32 - the serialization form every arm's checkpoint carries
(bit-equality with the F64 D-9 hash is impossible in this form by
construction; the F32 round-trip hash is a5557279...); (c) the
unnormalizer's action stats equal the pooled action stats (F32).

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
POST_STATE = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

import shutil

copied = []
for src in sorted(POOLED.iterdir()):
    if src.name.startswith("policy_preprocessor"):
        shutil.copy2(src, ckpt / src.name)
        copied.append(src.name)
print("copied pooled preprocessor files:", copied)
assert "policy_preprocessor.json" in copied
# The preprocessor references a tokenizer artifact directory; the
# substrate's save_model_checkpoint does not copy it into round
# checkpoints, so packaging must add it (else the packaged dir
# fails to load: "Missing processor artifact 'tokenizer'").
post_src = POOLED / POST_STATE
assert post_src.is_file(), post_src
shutil.copy2(post_src, ckpt / POST_STATE)
print("copied postprocessor state:", POST_STATE)
tok_src = POOLED / "tokenizer"
if tok_src.is_dir():
    tok_dst = ckpt / "tokenizer"
    if tok_dst.exists():
        shutil.rmtree(tok_dst)
    shutil.copytree(tok_src, tok_dst)
    print("copied tokenizer artifact dir")

import fl_driver
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained(str(ckpt))
pre, post = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=str(ckpt)
)
h = fl_driver.preprocessor_stats_hash(pre)
print("packaged preprocessor stats hash (F32 serialization):", h)
# The packaged form is lerobot's F32 serialization - the same form
# every arm's checkpoint carries - so bit-equality with the F64
# D-9 hash is impossible BY CONSTRUCTION (measured: the F32
# round-trip hash is a5557279...). The binding acceptance is
# tensor-level: every packaged mean/std equals the pooled D-9
# stats cast to F32, exactly.
import numpy as np
from safetensors.torch import load_file as _lf

pooled = _lf(str(POOLED / "pooled_stats.safetensors"))
norm = None
for step in pre.steps:
    if type(step).__name__ == "NormalizerProcessorStep":
        norm = step
        break
assert norm is not None
for key in sorted(norm.stats):
    for sub in ("mean", "std"):
        if sub not in norm.stats[key]:
            continue
        v = norm.stats[key][sub]
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        got = np.asarray(v, dtype=np.float64)
        want64 = pooled[f"{key}|{sub}"].numpy().astype(np.float32).astype(np.float64)
        # The F32 serialization squeezes scalar stats ((1,) -> ());
        # values and order are the binding content.
        assert got.size == want64.size and np.array_equal(
            got.reshape(-1), want64.reshape(-1)
        ), f"packaged stats mismatch at {key}.{sub}"
print("packaged preprocessor stats == pooled D-9 stats at F32: PASS")

# The unnormalizer must carry the pooled action stats (F32 form).
import torch
from safetensors.torch import load_file

want = load_file(str(POOLED / POST_STATE))
found = None
for step in post.steps:
    st = getattr(step, "stats", None)
    if isinstance(st, dict) and "action" in st:
        found = st
        break
assert found is not None, "unnormalizer stats not found in loaded postprocessor"
for sub in ("mean", "std"):
    got = torch.as_tensor(found["action"][sub]).detach().cpu().to(torch.float32)
    assert torch.equal(got, want[f"action.{sub}"]), f"post action.{sub} mismatch"
print("packaged postprocessor action stats == pooled (F32) PASS")
print("PACKAGING PASS:", ckpt)
