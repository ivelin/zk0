"""Study 001 prep (D-11/G-6, adaptation #9): build the ONE pooled
union-250 preprocessor of deviation D-9 with the study driver's own
construction (fl_driver.build_pooled_preprocessor, unmodified) and
gate on the driver's preprocessor_stats_hash == the D-9 recorded
hash. Then serialize for the substrate:

- the pipeline itself via save_pretrained (the packaging form,
  G-8; lerobot's round-trip degrades stats to float32 - measured
  max diff 1.8e-6 - so this form is NOT what clients train with);
- the normalizer statistics bit-exactly as float64 safetensors
  (pooled_stats.safetensors), from which every client reconstructs
  the pipeline through the driver's own factory call.

Final gate: construct the preprocessor through the substrate's
REAL client path (training_setup.build_batch_preprocessor, pooled
mode) and require the driver's hash over THAT object == the D-9
value; also unit-check the study rename collate.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-20261119/run/scripts")
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-fl-20261121")

EXPECTED = "a0455b5ba7d75b9c3d743754828020ca751f29fe93447a106207553b89d7a39d"
OUT = Path(
    "/home/openclaw/muse-work/zk0-study001-fl-20261121/study_assets/pooled_preprocessor"
)

import fl_driver

pre, h = fl_driver.build_pooled_preprocessor()
print("built stats hash:", h, flush=True)
assert h == EXPECTED, f"stats hash mismatch: {h} != {EXPECTED}"

OUT.mkdir(parents=True, exist_ok=True)
pre.save_pretrained(str(OUT), config_filename="policy_preprocessor.json")
print("pipeline serialization saved:", sorted(p.name for p in OUT.iterdir()), flush=True)

# The packaged (eval-facing) form follows the study's train-saved
# convention (cf. the centralized arm's checkpoint): the rename
# step carries the registered RENAME map, so a consumer feeding
# raw ArmnetBench batches gets the rename inside the pipeline -
# the same net transformation the driver applies in its data path
# during training. (The driver's factory call passes the rename as
# an `overrides=` kwarg this lerobot version silently ignores, so
# the raw saved json carries {}; the stats - the gated content -
# are unaffected.)
import json as _json

_cfg_path = OUT / "policy_preprocessor.json"
_cfg = _json.loads(_cfg_path.read_text())
for _step in _cfg["steps"]:
    if _step.get("registry_name") == "rename_observations_processor":
        _step["config"]["rename_map"] = fl_driver.RENAME
_cfg_path.write_text(_json.dumps(_cfg, indent=2))
print("packaged preprocessor json: rename_map set to RENAME", flush=True)

# ---- bit-exact stats serialization ----
import torch
from safetensors.torch import save_file

norm = None
for step in pre.steps:
    if type(step).__name__ == "NormalizerProcessorStep":
        norm = step
        break
assert norm is not None
flat = {}
for key in sorted(norm.stats):
    for sub, v in norm.stats[key].items():
        t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        flat[f"{key}|{sub}"] = t.detach().cpu().clone()
save_file(flat, str(OUT / "pooled_stats.safetensors"), metadata={"format": "pt"})
print(f"pooled stats serialized: {len(flat)} tensors", flush=True)

# Postprocessor state for packaging (G-8): the unnormalizer step's
# state file in lerobot's own save format (keys "key.sub", F32) -
# the substrate's save_model_checkpoint does not write it, and a
# packaged checkpoint does not load without it. Content: the same
# pooled stats (the unnormalizer reads the action entries).
post_flat = {
    f"{key}.{sub}": torch.as_tensor(t).to(torch.float32).contiguous()
    for key in sorted(norm.stats)
    for sub, t in ((s, norm.stats[key][s]) for s in norm.stats[key])
}
save_file(
    post_flat,
    str(OUT / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"),
    metadata={"format": "pt"},
)
print(f"postprocessor state serialized: {len(post_flat)} tensors", flush=True)

# ---- end-to-end gate through the real client construction path ----
from src.common.utils import load_lerobot_dataset
from src.client_app import _load_episode_list, _study_rename_collate
from src.configs import DatasetConfig
from src.training.model_utils import get_model
from src.training.training_setup import build_batch_preprocessor

cfg = DatasetConfig.load()
c0 = sorted(cfg.clients, key=lambda c: c.client_id)[0]
ds0 = load_lerobot_dataset(
    c0.name, root=c0.root, episodes=_load_episode_list(c0.episodes_file)
)
policy = get_model(dataset_meta=ds0.meta)
constructed = build_batch_preprocessor(policy, ds0)
h2 = fl_driver.preprocessor_stats_hash(constructed)
print("client-path constructed stats hash:", h2, flush=True)
assert h2 == EXPECTED, f"client-path stats hash mismatch: {h2} != {EXPECTED}"

# ---- rename collate unit check ----
fake = [
    {
        "observation.images.front": torch.zeros(3, 4, 4),
        "observation.images.top": torch.ones(3, 4, 4),
        "observation.images.wrist": torch.full((3, 4, 4), 2.0),
        "action": torch.zeros(6),
        "task": "stack the blocks",
    }
]
batch = _study_rename_collate(fake)
assert "observation.images.camera1" in batch, batch.keys()
assert "observation.images.camera2" in batch and "observation.images.camera3" in batch
assert "observation.images.front" not in batch
print("rename collate check PASS", flush=True)
print("POOLED_PREPROCESSOR_GATE PASS")
