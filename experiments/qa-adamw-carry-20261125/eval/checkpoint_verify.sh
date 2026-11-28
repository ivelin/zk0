#!/usr/bin/env bash
# checkpoint_verify.sh — shared verifier for the checkpoint + resume standard
# (Ivelin, 2026-11-18, after the FT-base loss).
#
# A weights-only checkpoint is NOT a resume. This script checks a checkpoint
# directory for FULL state, hashes every file, verifies integrity against a
# prior resume manifest when one exists, attempts a reload of any tensor
# files whose loaders are available, and writes a fresh resume manifest.
# Training and eval pipelines call this instead of re-implementing the check.
#
# Usage: checkpoint_verify.sh <checkpoint-dir>
#   Required components (convention-based, searched recursively to depth 3):
#     weights    *.safetensors | *.bin | *.pt | *.pth
#     optimizer  any file matching *optim*
#     scheduler  any file matching *sched*
#     rng        any file matching *rng*
#     step       trainer_state.json | any file matching *step*
#   CHECKPOINT_REQUIRE (env): space-separated subset to require something
#     else, e.g. CHECKPOINT_REQUIRE="weights step" for an eval snapshot.
#   The manifest is <checkpoint-dir>/resume-manifest.json. If one already
#     exists, files recorded in it must still hash identically — a changed or
#     missing recorded file is an INTEGRITY FAIL and the manifest is left
#     untouched. Extra unrecorded files are noted, not fatal.
#   Reload: attempted with torch (for .pt/.pth) / safetensors when the
#     library imports; otherwise recorded as "skipped (library unavailable)".
# Exit: 0 = complete + verified, 1 = incomplete / integrity / reload failure.
set -uo pipefail

DIR=${1:?usage: checkpoint_verify.sh <checkpoint-dir>}
[ -d "$DIR" ] || { echo "checkpoint_verify: not a directory: $DIR" >&2; exit 1; }
REQUIRE=${CHECKPOINT_REQUIRE:-"weights optimizer scheduler rng step"}
MANIFEST="$DIR/resume-manifest.json"
export CKPT_DIR="$DIR" CKPT_REQUIRE="$REQUIRE" CKPT_MANIFEST="$MANIFEST"

python3 - <<'PYEOF'
import hashlib, json, os, sys, datetime

d = os.environ["CKPT_DIR"]; require = os.environ["CKPT_REQUIRE"].split()
manifest_path = os.environ["CKPT_MANIFEST"]

def find(pred, maxdepth=3):
    out = []
    for root, dirs, files in os.walk(d):
        depth = root[len(d):].count(os.sep)
        if depth >= maxdepth: dirs[:] = []
        for f in files:
            if f == "resume-manifest.json": continue
            if pred(f): out.append(os.path.relpath(os.path.join(root, f), d))
    return sorted(out)

def is_weights(f):
    low = f.lower()
    if any(k in low for k in ("optim", "sched", "rng")): return False
    return f.endswith((".safetensors", ".bin", ".pt", ".pth"))

components = {
    "weights":   find(is_weights),
    "optimizer": find(lambda f: "optim" in f.lower()),
    "scheduler": find(lambda f: "sched" in f.lower()),
    "rng":       find(lambda f: "rng" in f.lower()),
    "step":      find(lambda f: f == "trainer_state.json" or "step" in f.lower()),
}
missing = [c for c in require if not components.get(c)]
for c in sorted(components):
    mark = "ok " if components[c] else "MISS"
    print(f"component {c:9s} [{mark}] {components[c] if components[c] else ''}")

# hash every file
hashes = {}
for root, dirs, files in os.walk(d):
    for f in files:
        if f == "resume-manifest.json": continue
        p = os.path.join(root, f); rel = os.path.relpath(p, d)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
        hashes[rel] = {"sha256": h.hexdigest(), "size": os.path.getsize(p)}

# integrity against a prior manifest
integrity = "no prior manifest"
fail = bool(missing)
if os.path.exists(manifest_path):
    prior = json.load(open(manifest_path))
    bad = []
    for rel, rec in prior.get("files", {}).items():
        if rel not in hashes: bad.append(f"{rel}: missing")
        elif hashes[rel]["sha256"] != rec["sha256"]: bad.append(f"{rel}: hash changed")
    extra = [r for r in hashes if r not in prior.get("files", {})]
    if bad:
        integrity = "FAIL: " + "; ".join(bad); fail = True
        print("INTEGRITY " + integrity)
    else:
        integrity = "OK (all recorded files hash-identical)"
        print("INTEGRITY OK" + (f"; note: {len(extra)} unrecorded extra file(s): {extra}" if extra else ""))

# reload attempt
reload_res = {}
tensor_files = [r for r in hashes if r.endswith((".pt", ".pth", ".safetensors"))]
if tensor_files:
    try:
        import torch
        for rel in tensor_files:
            if rel.endswith((".pt", ".pth")):
                try: torch.load(os.path.join(d, rel), map_location="cpu"); reload_res[rel] = "loaded (torch)"
                except Exception as e: reload_res[rel] = f"FAILED (torch): {e}"; fail = True
    except ImportError:
        pass
    try:
        from safetensors import safe_open
        for rel in tensor_files:
            if rel.endswith(".safetensors") and rel not in reload_res:
                try:
                    with safe_open(os.path.join(d, rel), framework="pt") as fh: list(fh.keys())
                    reload_res[rel] = "loaded (safetensors)"
                except Exception as e: reload_res[rel] = f"FAILED (safetensors): {e}"; fail = True
    except ImportError:
        pass
    for rel in tensor_files:
        reload_res.setdefault(rel, "skipped (no loader library available)")
for rel, res in sorted(reload_res.items()): print(f"reload {rel}: {res}")

manifest = {
    "checkpoint": os.path.abspath(d),
    "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "require": require,
    "components": components,
    "missing": missing,
    "integrity": integrity,
    "reload": reload_res,
    "files": hashes,
    "resume_ready": not fail,
}
if fail and os.path.exists(manifest_path) and integrity.startswith("FAIL"):
    print("manifest NOT rewritten (integrity failure must be investigated first)")
else:
    with open(manifest_path, "w") as fh: json.dump(manifest, fh, indent=2)
    print(f"manifest written: {manifest_path}")
print("RESUME READY" if not fail else "NOT RESUME READY")
sys.exit(0 if not fail else 1)
PYEOF
