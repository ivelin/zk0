"""Twin-suite v2 policy service (runs in the FL venv: torch+lerobot).

Long-running process speaking JSON-lines on stdin/stdout. The sim runner
(MuJoCo 4.0.1 venv) sends one request per action chunk:
  {"state_deg": [6 floats], "task": str, "episode": int, "chunk": int,
   "frames": {"front": path.npy, "top": path.npy, "wrist": path.npy}}
Frames are uint8 HWC .npy renders (256x256x3) written by the sim runner.
Response: {"chunk_deg": [[6] x 50]}  (dataset units: degrees, post-processed
exactly as the Study 001 eval harness does).

Input contract mirrors run/eval-traj-20261122/scripts/eval_traj.py:
CAM_MAP front->camera1, top->camera2, wrist->camera3 (base config keys);
A5 rename-step drop; seeded noise per (episode, chunk): seed=epi*1000+chunk.
Device: CPU by choice (recorded in the Stage 1.4 report) -- DGX training
must not be disturbed; SmolVLA chunk inference on CPU is acceptable for a
mechanics smoke and for gate runs is a scheduling decision, not a contract
change.
"""
import json, os, sys
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def main():
    ckpt = sys.argv[1]
    import torch
    torch.set_num_threads(int(os.environ.get("POLICY_THREADS", "4")))
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    policy = SmolVLAPolicy.from_pretrained(ckpt)
    policy.to("cpu").eval()
    policy.config.device = "cpu"
    try:
        pre, post = make_pre_post_processors(
            policy.config, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
    except Exception:
        pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt)
    # Post-hoc device pin (the base checkpoint's processor config declares
    # cuda; this service is CPU-only by design).
    for pipe in (pre, post):
        for st in getattr(pipe, "steps", []):
            if hasattr(st, "device"):
                try: st.device = "cpu"
                except Exception: pass
            if hasattr(st, "tensor_device"):
                try: st.tensor_device = torch.device("cpu")
                except Exception: pass
    # A5 fix (identical to eval_traj.py): drop rename steps that rename AWAY
    # from the policy's declared features.
    try:
        want = set(policy.config.image_features.keys()) | {"observation.state"}
        kept = []
        for st in pre.steps:
            rn = getattr(st, "rename_map", None)
            if rn and any(k in want and v not in want for k, v in rn.items()):
                continue
            kept.append(st)
        pre.steps = kept
    except Exception as e:
        print(json.dumps({"warn": f"rename filter skipped: {e}"}), flush=True)
    cam_map = {"front": "observation.images.camera1",
               "top": "observation.images.camera2",
               "wrist": "observation.images.camera3"}
    # If the checkpoint declares dataset camera keys instead, map identically.
    keys = set(policy.config.image_features.keys())
    if "observation.images.front" in keys:
        cam_map = {k: f"observation.images.{k}" for k in cam_map}
    print("READY " + json.dumps({"ready": True, "cam_map": cam_map,
                      "chunk_size": int(policy.config.chunk_size)}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "quit":
            break
        batch = {}
        for ds_key, pol_key in cam_map.items():
            img = np.load(req["frames"][ds_key])  # uint8 HWC
            t = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
            batch[pol_key] = t
        batch["observation.state"] = torch.tensor(
            [req["state_deg"]], dtype=torch.float32)
        batch["task"] = [req["task"]]
        batch = pre(batch)
        g = torch.Generator(device="cpu").manual_seed(
            int(req["episode"]) * 1000 + int(req["chunk"]))
        noise = torch.randn((1, int(policy.config.chunk_size),
                             policy.config.max_action_dim),
                            generator=g, dtype=batch["observation.state"].dtype)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch, noise=noise)
        try:
            act = post({"action": chunk})["action"]
        except Exception:
            act = post(chunk)
        act = act[0].float().cpu().numpy()
        if act.shape[-1] > 6:
            act = act[:, :6]
        print(json.dumps({"chunk_deg": act.tolist()}), flush=True)

if __name__ == "__main__":
    main()
