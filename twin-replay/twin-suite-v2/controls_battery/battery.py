#!/usr/bin/env python3
"""Twin-suite v2 Stage 1 controls battery (Part B): scene generation + resumable worker.

Layers (controls-battery-prep-v2.md S4, CERTIFIED):
  L1 nominal MuJoCo: 20 validated scenes x K=5 variants (k=0..4; k=0 is the
     frozen empty-draw nominal instance) x 3 checkpoints = 300
  L2 envelope corners MuJoCo: 5 envelope-validated precision scenes x 8
     non-nominal corners x 3 checkpoints = 120 (nominal instance)
  L3 Drake nominal: same 5 scenes x 3 checkpoints = 15
All MuJoCo scenes are v1.1 scene copies with ONLY the arm include swapped
to the padded v2 arm (the reference controllers' ensure_scene convention);
no v1.1 file is modified. Corner scenes additionally apply exactly one
corner deviation per envelope-corners-v2.json.
"""
import json, re, shutil, subprocess, sys, time
from pathlib import Path

V1 = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
V2 = Path("/home/openclaw/muse-work/zk0-study001-twin-v2-20261122")
BAT = V2 / "controls_battery"
SCENES = BAT / "scenes-battery"
OUT = BAT / "out"
SIM_PY = "/home/openclaw/muse-work/zk0-sim-v01-20261115/.venv/bin/python"
DK_PY = "/home/openclaw/muse-work/zk0-sim-eval2-20261115/drake/venv/bin/python3"
PADDED = str(V2 / "scenes-v2/arm/so101_new_calib_padded.xml")
ORIG_INCLUDE = "/home/openclaw/muse-work/zk0-sim-v01-20261115/sim/model/so101_new_calib.xml"

VALIDATED = {
    "block_stack": [411, 453, 489, 498, 499, 527, 546, 549],
    "eye_drops_to_basket": [1630, 1746, 1798, 1609],
    "tool_insert": [683],
    "cable_clip": [1969, 2020, 2130, 1905, 2046, 2068, 2085],
}
ENVELOPE_PRECISION = {"block_stack": [499, 527, 546, 549], "eye_drops_to_basket": [1609]}
MANIPULATED = {"block_stack": ["cube_a", "cube_b"], "eye_drops_to_basket": ["carton"],
               "tool_insert": ["tool"]}
CORNERS = ["pose_+x", "pose_-x", "pose_+y", "pose_-y", "friction_low", "friction_high",
           "mass_low", "mass_high"]
CKPTS = [  # id, path -- from controls/controls-manifest.json (pinned, Part A)
    ("base", "/home/openclaw/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"),
    ("fl_round_30_a5repair", "/home/openclaw/muse-work/zk0-study001-fl-20261121/outputs/2026-11-21_14-56-47/models/checkpoint_round_30_a5repair"),
    ("c_exp", "/home/openclaw/muse-work/zk0-study001-20261119/run/train/centralized/out/checkpoints/015000/pretrained_model"),
]

def padded_copy(src: Path, dst: Path):
    txt = src.read_text()
    assert ORIG_INCLUDE in txt, f"include not found in {src}"
    dst.write_text(txt.replace(ORIG_INCLUDE, PADDED))

def gen_scenes():
    SCENES.mkdir(parents=True, exist_ok=True)
    corners_doc = json.loads((BAT / "envelope-corners-v2.json").read_text())
    n = 0
    for task, eps in VALIDATED.items():
        for ep in eps:
            base = V1 / "scenes" / f"{task}__ep{ep}.xml"
            padded_copy(base, SCENES / base.name); n += 1
            shutil.copy(V1 / "scenes" / f"{task}__ep{ep}_variants.json",
                        SCENES / f"{task}__ep{ep}_variants.json")
            for k in range(5):
                v = V1 / "scenes" / f"{task}__ep{ep}_v{k}.xml"
                padded_copy(v, SCENES / v.name); n += 1
    # corners: from the padded nominal copies
    for task, eps in ENVELOPE_PRECISION.items():
        tc = corners_doc["tasks"][task]
        for ep in eps:
            nominal = (SCENES / f"{task}__ep{ep}.xml").read_text()
            for corner in CORNERS:
                txt = apply_corner(nominal, task, corner, tc, corners_doc)
                (SCENES / f"{task}__ep{ep}__{corner}.xml").write_text(txt); n += 1
    print(f"generated {n} scene files")

def _geom_key(name, task, tc, doc):
    if name == "floor": return doc["global_friction_sliding"]["floor"]
    if name == "mat": return doc["global_friction_sliding"]["mat"]
    if name.startswith("wall"): return doc["global_friction_sliding"]["walls"]
    fs = tc.get("friction_sliding", {})
    for key in fs:
        if key in name: return fs[key]
    return None

def apply_corner(txt, task, corner, tc, doc):
    cdef = next(c for c in tc["corners"] if c["id"] == corner)
    if cdef["type"] == "pose":
        dx, dy = cdef["offset_m"]["x"], cdef["offset_m"]["y"]
        for body in MANIPULATED[task]:
            def repl(m, dx=dx, dy=dy):
                x, y, z = (float(v) for v in m.group(2).split())
                return f'{m.group(1)}{x+dx:.6f} {y+dy:.6f} {z:g}"'
            txt = re.sub(r'(<body name="' + body + r'" pos=")([^"]+)"', repl, txt, count=1)
    elif cdef["type"] == "friction":
        lvl = cdef["setting"]
        def grepl(m):
            name, attrs = m.group(1), m.group(2)
            ent = _geom_key(name, task, tc, doc)
            if ent is None: return m.group(0)
            def frepl(fm, ent=ent, lvl=lvl):
                parts = fm.group(1).split()
                parts[0] = str(ent[lvl])
                return 'friction="' + " ".join(parts) + '"'
            return "<geom name=\"" + name + "\"" + re.sub(r'friction="([^"]+)"', frepl, attrs) + m.group(3) + ">"
        txt = re.sub(r'<geom name="([^"]+)"([^>]*?)(/?)>', grepl, txt)
    elif cdef["type"] == "mass":
        lvl = cdef["setting"]
        for body in MANIPULATED[task]:
            mass = tc["mass_kg"][body][lvl]
            pat = re.compile(r'(<body name="' + body + r'"[^>]*>.*?<geom[^>]*?)mass="[^"]+"')
            txt = pat.sub(lambda m: m.group(1) + f'mass="{mass}"', txt, count=1)
    return txt

def worklist():
    units = []
    for ckid, ckpath in CKPTS:
        for task, eps in VALIDATED.items():
            for ep in eps:
                for k in range(5):
                    units.append({"layer": "L1", "checkpoint_id": ckid, "checkpoint_path": ckpath,
                                  "task": task, "episode": ep, "variant": str(k), "corner": "nominal",
                                  "engine": "mujoco", "out": str(OUT / "L1" / ckid)})
        for task, eps in ENVELOPE_PRECISION.items():
            for ep in eps:
                for corner in CORNERS:
                    units.append({"layer": "L2", "checkpoint_id": ckid, "checkpoint_path": ckpath,
                                  "task": task, "episode": ep, "variant": "nominal", "corner": corner,
                                  "engine": "mujoco", "out": str(OUT / "L2" / ckid),
                                  "scene_file": str(SCENES / f"{task}__ep{ep}__{corner}.xml")})
                units.append({"layer": "L3", "checkpoint_id": ckid, "checkpoint_path": ckpath,
                              "task": task, "episode": ep, "variant": "nominal", "corner": "nominal",
                              "engine": "drake", "out": str(OUT / "L3" / ckid)})
    (BAT / "worklist.jsonl").write_text("\n".join(json.dumps(u) for u in units) + "\n")
    print(f"worklist: {len(units)} units")
    return units

def recorded(unit):
    p = Path(unit["out"]) / "rollouts.jsonl"
    if not p.exists(): return False
    for line in p.read_text().splitlines():
        try: r = json.loads(line)
        except Exception: continue
        if (r.get("task") == unit["task"] and r.get("episode") == unit["episode"]
                and str(r.get("variant")) == unit["variant"]
                and r.get("checkpoint_id") == unit["checkpoint_id"]
                and r.get("status") == "complete"
                and (unit["engine"] == "drake" or r.get("corner", "nominal") == unit["corner"])):
            return True
    return False

def update_manifest(units):
    done = {}
    for u in units:
        k = f'{u["layer"]}/{u["checkpoint_id"]}'
        done.setdefault(k, [0, 0]); done[k][1] += 1
        if recorded(u): done[k][0] += 1
    man = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "total_units": len(units),
           "completed_units": sum(v[0] for v in done.values()),
           "by_layer_checkpoint": {k: {"completed": v[0], "total": v[1]} for k, v in sorted(done.items())}}
    (BAT / "completion-manifest.json").write_text(json.dumps(man, indent=1))
    return man

def run_unit(u):
    out = Path(u["out"]); out.mkdir(parents=True, exist_ok=True)
    if u["engine"] == "mujoco":
        cmd = ["nice", "-n", "10", SIM_PY, str(V2 / "harness/sim_runner_battery.py"),
               "--task", u["task"], "--episode", str(u["episode"]), "--variant", u["variant"],
               "--checkpoint-id", u["checkpoint_id"], "--checkpoint-path", u["checkpoint_path"],
               "--policy", "service", "--out", u["out"], "--save-frames",
               "--corner", u["corner"]]
        if "scene_file" in u: cmd += ["--scene-file", u["scene_file"]]
        cwd = V2
    else:
        cmd = ["nice", "-n", "19", DK_PY, "sim_runner_drake.py",
               "--task", u["task"], "--episode", str(u["episode"]), "--variant", "nominal",
               "--checkpoint-id", u["checkpoint_id"], "--checkpoint-path", u["checkpoint_path"],
               "--policy", "service", "--out", u["out"], "--save-frames"]
        cwd = V2 / "drake_actuated"
    log = BAT / "logs" / f'{u["layer"]}_{u["checkpoint_id"]}_{u["task"]}_{u["episode"]}_{u["variant"]}_{u["corner"]}.log'
    with open(log, "a") as lf:
        lf.write(f"\n=== {time.ctime()} {cmd}\n"); lf.flush()
        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        if u["engine"] == "drake": env["DK_CONTACT"] = "hydro"
        r = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, timeout=1800, env=env)
    return r.returncode

def work(stream):
    units = [json.loads(l) for l in (BAT / "worklist.jsonl").read_text().splitlines()]
    if stream == "drake":
        mine = [u for u in units if u["engine"] == "drake"]
    else:
        mine = [u for u in units if u["engine"] == "mujoco" and u["checkpoint_id"] == stream]
    print(f"stream {stream}: {len(mine)} units", flush=True)
    for i, u in enumerate(mine):
        if recorded(u):
            print(f"[{stream}] skip recorded {u['layer']} {u['task']} {u['episode']} {u['variant']} {u['corner']}", flush=True); continue
        t0 = time.time()
        try: rc = run_unit(u)
        except subprocess.TimeoutExpired: rc = "TIMEOUT"
        ok = recorded(u)
        print(f"[{stream}] {i+1}/{len(mine)} {u['layer']} {u['task']} ep{u['episode']} v{u['variant']} {u['corner']} rc={rc} recorded={ok} {time.time()-t0:.0f}s", flush=True)
        update_manifest(units)
    print(f"stream {stream} DONE", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gen": gen_scenes(); worklist()
    elif mode == "work": work(sys.argv[2])
    elif mode == "manifest":
        units = [json.loads(l) for l in (BAT / "worklist.jsonl").read_text().splitlines()]
        print(json.dumps(update_manifest(units), indent=1))
