"""Study 001 pre-flight gates for the FL re-execution (D-11):
1. Resolved smolvla_base snapshot == pinned revision, and its
   model.safetensors SHA-256 == the sign-off value.
2. Trainable census on the policy built through the substrate's own
   get_model path == certified 155 tensors / 99,880,992 params
   (500 tensors total).
3. Per-client dataset wiring (adaptation #4): each study client
   loads exactly 50 episodes and the P0 manifest frame count, from
   the pinned on-disk tree; dataset fps == 20; policy chunk == 50.
Run from the study copy root with the study venv.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-fl-20261121")

PINNED_SNAPSHOT = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
PINNED_MODEL_SHA256 = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
W = Path("/home/openclaw/muse-work/zk0-study001-20261119")

# ---- Gate 1: snapshot resolution ----
from huggingface_hub import snapshot_download

snap = Path(snapshot_download("lerobot/smolvla_base"))
print("resolved snapshot:", snap)
assert snap.name == PINNED_SNAPSHOT, f"snapshot drift: {snap.name} != {PINNED_SNAPSHOT}"
sha = hashlib.sha256((snap / "model.safetensors").read_bytes()).hexdigest()
print("model.safetensors sha256:", sha)
assert sha == PINNED_MODEL_SHA256, "model.safetensors sha mismatch"
print("GATE 1 PASS: pinned snapshot + weights hash")

# ---- Gates 2+3 need the substrate loaders ----
from src.common.utils import load_lerobot_dataset
from src.configs import DatasetConfig
from src.client_app import _load_episode_list

manifest = json.loads((W / "manifest.json").read_text())
frames = {t: s["total_frames"] for t, s in manifest["subsets"].items()}
node_task = {
    int(k.split("_")[1]): v["task_id"] for k, v in manifest["fl_node_assignment"].items()
}

cfg = DatasetConfig.load()
first_meta = None
for client in sorted(cfg.clients, key=lambda c: c.client_id):
    task = node_task[client.client_id]
    episodes = _load_episode_list(client.episodes_file)
    assert len(episodes) == 50, f"{task}: {len(episodes)} episodes in file"
    assert episodes == manifest["subsets"][task]["episode_indices"], (
        f"{task}: episodes file != P0 manifest episode list"
    )
    ds = load_lerobot_dataset(client.name, root=client.root, episodes=episodes)
    n_eps = ds.num_episodes if hasattr(ds, "num_episodes") else len(ds.meta.episodes)
    n_frames = len(ds)
    print(f"client {client.client_id} ({task}): episodes={n_eps} frames={n_frames} fps={ds.meta.fps}")
    assert n_eps == 50, f"{task}: loaded {n_eps} episodes"
    assert n_frames == frames[task], f"{task}: loaded {n_frames} frames != {frames[task]}"
    assert int(ds.meta.fps) == 20, f"{task}: fps {ds.meta.fps} != 20"
    if first_meta is None:
        first_meta = ds.meta
print("GATE 3 PASS: all 5 clients load their P0 subsets (50 eps, manifest frames, fps 20)")

# ---- Gate 2: census through the substrate construction ----
from src.training.model_utils import get_model

policy = get_model(dataset_meta=first_meta)
total = [p for p in policy.parameters()]
trainable = [p for p in policy.parameters() if p.requires_grad]
n_total, n_train = len(total), len(trainable)
n_train_params = sum(p.numel() for p in trainable)
print(f"census: total tensors={n_total} trainable tensors={n_train} trainable params={n_train_params}")
assert n_total == 500, n_total
assert n_train == 155, n_train
assert n_train_params == 99_880_992, n_train_params
chunk = policy.config.chunk_size
print("policy chunk_size:", chunk)
assert int(chunk) == 50, chunk
print("GATE 2 PASS: census 155 / 99,880,992 of 500; chunk 50")
print("ALL PREFLIGHT GATES PASS")
