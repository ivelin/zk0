"""Re-run v1.1 MuJoCo replay (nominal variant) capturing final quaternions.

v1.1's stored validation results carry final positions only; the v2
equivalence diagnostics need orientations too. This wrapper monkeypatches
replay.evaluate (in-process only; NO v1.1 file is modified) to append
objects_final_quat_wxyz, then calls the unmodified run_episode.
"""
import json, sys
from pathlib import Path
import numpy as np

V1 = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
sys.path.insert(0, str(V1 / "code"))
import replay as R

_orig = R.evaluate

def _wrapped(model, data, task, epi, mujoco):
    res = _orig(model, data, task, epi, mujoco)
    quats = {}
    for nm in res.get("objects_final", {}):
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            quats[nm] = np.array(data.xquat[bid]).tolist()
        except Exception:
            pass
    res["objects_final_quat_wxyz"] = quats
    res["engine"] = "mujoco-rerun"
    return res

R.evaluate = _wrapped

if __name__ == "__main__":
    task, epi = sys.argv[1], int(sys.argv[2])
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("mujoco_results")
    r = R.run_episode(task, epi, variant=0, out_dir=str(out))
    print(json.dumps({"task": task, "episode": epi, "engine": "mujoco",
                      "sim_success": r["sim_success"]}, default=str))
