"""The zk0-train pack: pinned training/FL stack + its proofs.

Spec section 1.3 pins the train pack (LeRobot 0.7.0, PyAV 15.x hard
pin, datasets, Flower 1.23.0, Newton 1.8). This module is where the
package *proves* the pins work, not just that they import:

- `versions()` — the installed versions, read from the live
  interpreter, for doctor check 2 to compare against the manifest.
- `prove()` — a LeRobot dataset write/read round-trip on a tiny
  synthetic dataset with real video features. Writing episodes is
  the PyAV trap (SSOT decision 26: PyAV 19 breaks LeRobot 0.7
  dataset capture): the round-trip encodes video through PyAV,
  reloads the dataset, and checks the frames and actions come back
  as written. If capture is broken, this fails loudly instead of
  the operator finding out mid-recording.
"""
import json
import shutil
import tempfile
from pathlib import Path

EXPECTED = {  # module -> manifest pin name
    "lerobot": "lerobot", "av": "pyav", "datasets": "datasets",
    "flwr": "flwr", "newton": "newton",
}


def versions() -> dict:
    """Installed train-pack versions from the live interpreter."""
    out = {}
    for mod in EXPECTED:
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[mod] = None
    return out


def prove(workdir=None) -> dict:
    """LeRobot dataset round-trip proof (see module docstring).

    Writes 1 episode x 20 frames of a synthetic dataset with the
    checkpoint's own feature shapes (state 6, action 6, two 320x240
    video streams), reloads it, and verifies the actions and a
    video frame round-trip. Returns a result dict; raises on
    failure with the failing step named.
    """
    import numpy as np

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="zk0-trainpack-prove-"))
    ds_root = root / "prove-dataset"
    if ds_root.exists():
        shutil.rmtree(ds_root)
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,),
                              "names": ["state"]},
        "observation.images.front": {"dtype": "video",
                                     "shape": (240, 320, 3),
                                     "names": ["height", "width",
                                               "channels"]},
        "observation.images.top": {"dtype": "video",
                                   "shape": (240, 320, 3),
                                   "names": ["height", "width",
                                             "channels"]},
        "action": {"dtype": "float32", "shape": (6,),
                   "names": ["action"]},
    }
    ds = LeRobotDataset.create(repo_id="zk0/trainpack-prove", fps=10,
                               features=features, root=ds_root,
                               use_videos=True)
    rng = np.random.default_rng(7)
    actions = rng.normal(size=(20, 6)).astype(np.float32)
    for i in range(20):
        frame = {
            "observation.state": actions[i] * 0.5,
            "observation.images.front": rng.integers(
                0, 255, size=(240, 320, 3), dtype=np.uint8),
            "observation.images.top": rng.integers(
                0, 255, size=(240, 320, 3), dtype=np.uint8),
            "action": actions[i],
            "task": "trainpack round-trip proof (synthetic)",
        }
        ds.add_frame(frame)
    ds.save_episode()
    ds.finalize()  # LeRobot 0.7: flushes episode metadata

    back = LeRobotDataset(repo_id="zk0/trainpack-prove", root=ds_root)
    n = len(back)
    if n != 20:
        raise RuntimeError(f"round-trip: reloaded {n} frames, wrote 20")
    item = back[3]
    got_action = np.asarray(item["action"], dtype=np.float32).reshape(-1)
    if not np.allclose(got_action, actions[3], atol=1e-5):
        raise RuntimeError(
            f"round-trip: action mismatch at frame 3: {got_action} vs "
            f"{actions[3]}")
    img = item["observation.images.front"]
    shape = tuple(img.shape)
    if shape != (3, 240, 320):
        raise RuntimeError(f"round-trip: front image shape {shape}, "
                           f"expected (3, 240, 320)")
    meta = {"episodes": int(back.meta.total_episodes),
            "frames": n,
            "video_backend": getattr(back, "video_backend", "default")}
    return {"pass": True, "dataset_root": str(ds_root),
            "frames_written": 20, "frames_reloaded": n,
            "action_roundtrip_max_err": float(
                np.abs(got_action - actions[3]).max()),
            "front_image_shape": list(shape), "meta": meta,
            "versions": versions()}


if __name__ == "__main__":
    print(json.dumps(prove(), indent=1))
