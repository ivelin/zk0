#!/usr/bin/env bash
# Dev bootstrap for build slice 1 (Spark build host): create the
# package runtime (its own Python 3.12 venv + the slice-1 pins) and
# stage the pinned sim assets from the certified local copies,
# md5-verified against runtime-manifest.json.
#
# The product installer (spec §1.2: signed artifacts, hash-checked
# pins, atomic versions) is later build work; this script is the
# developer equivalent and performs the same hash checks.
set -euo pipefail
NODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON312:-python3.12}"

"$PY" -m venv "$NODE_DIR/runtime/venv"
"$NODE_DIR/runtime/venv/bin/pip" install --quiet "mujoco==4.0.1" "numpy==2.2.6"

ZK0_HOME="${ZK0_HOME:-$HOME/.zk0-node}"
mkdir -p "$ZK0_HOME/assets"

# Staging sources: the certified local copies on the build host.
# Override with ZK0_ASSETS_SRC_REALISTIC / ZK0_ASSETS_SRC_STOCK.
SRC_REALISTIC="${ZK0_ASSETS_SRC_REALISTIC:-/home/openclaw/muse-work/zk0-rung3-rematch-20261117/harness/model}"
SRC_STOCK="${ZK0_ASSETS_SRC_STOCK:-/home/openclaw/muse-work/zk0-multitask-20261115/model}"

rm -rf "$ZK0_HOME/assets/model-realistic" "$ZK0_HOME/assets/model-stock"
cp -r "$SRC_REALISTIC" "$ZK0_HOME/assets/model-realistic"
# The certified harness convention: the scene includes "so101.xml";
# for the realistic variant that file must BE so101_realistic.xml.
cp "$ZK0_HOME/assets/model-realistic/so101_realistic.xml" \
   "$ZK0_HOME/assets/model-realistic/so101.xml"
cp -r "$SRC_STOCK" "$ZK0_HOME/assets/model-stock"

check() { # file expected_md5 label
  got="$(md5sum "$1" | awk '{print $1}')"
  if [ "$got" != "$2" ]; then
    echo "bootstrap: HASH MISMATCH for $3: got $got, want $2" >&2; exit 1
  fi
  echo "bootstrap: $3 md5 $got OK"
}
check "$ZK0_HOME/assets/model-realistic/so101_realistic.xml" \
      8f3937f109fd512d027935b56689fff8 "so101-realistic canonical"
check "$ZK0_HOME/assets/model-realistic/so101.xml" \
      8f3937f109fd512d027935b56689fff8 "so101-realistic live include"
check "$ZK0_HOME/assets/model-stock/so101.xml" \
      1470b61558f0f18762e27e7431ac51ce "stock so101 (control)"

echo "bootstrap: runtime ready at $NODE_DIR/runtime/venv"
echo "bootstrap: assets staged at $ZK0_HOME/assets"

# ---------------------------------------------------------------- slice 2
# Train pack (spec §1.3): two phases. Phase 1 resolves LeRobot's
# stack against the certified environment's constraints
# (dev/trainpack-constraints.txt = that environment's freeze, minus
# the lines a named pin's own requirements supersede). Phase 2 adds
# Flower: flwr 1.23.0's click<8.2 is jointly unsatisfiable with
# huggingface-hub's click>=8.4.2 (every hub in LeRobot 0.7.0's
# range), so flwr's non-click dependencies install normally and
# flwr itself installs --no-deps. The override is recorded in
# runtime-manifest.json (flwr pin note); `pip check` will report
# exactly three conflicts, all flwr's — click<8.2 (carried: 8.5.1),
# iterators<0.0.3,>=0.0.2 (carried: 0.2.0), pathspec<0.13.0,>=0.12.1
# (carried: 1.1.1) — that is expected, disclosed, not a defect.
PIP="$NODE_DIR/runtime/venv/bin/pip"
"$PIP" install --quiet -c "$NODE_DIR/dev/trainpack-constraints.txt" \
  "lerobot[dataset,smolvla]==0.7.0" "av==15.1.0" "datasets==5.1.0" \
  "newton==1.8.0" "numpy==2.2.6" "mujoco==4.0.1"
"$PIP" install --quiet "grpcio" "grpcio-health-checking" "protobuf<5" \
  "cryptography<45" "pycryptodome<4" "iterators" "typer<0.13" "tomli" \
  "tomli-w" "pathspec" "rich<14"
"$PIP" install --quiet --no-deps "flwr==1.23.0"
echo "bootstrap: train pack installed (lerobot 0.7.0, av 15.1.0, datasets 5.1.0, flwr 1.23.0, newton 1.8.0)"

# SIM-2 object: the measured YCB mug package.
SRC_MUG="${ZK0_ASSETS_SRC_MUG:-/home/openclaw/muse-work/zk0-ycbmug-20261116}"
rm -rf "$ZK0_HOME/assets/ycb-025-mug"
mkdir -p "$ZK0_HOME/assets/ycb-025-mug"
cp "$SRC_MUG"/visual.obj "$SRC_MUG"/texture_map.png "$SRC_MUG"/params.json \
   "$ZK0_HOME/assets/ycb-025-mug/"
cp "$SRC_MUG"/hull_*.obj "$ZK0_HOME/assets/ycb-025-mug/"
check "$ZK0_HOME/assets/ycb-025-mug/visual.obj" \
      8ff104788e7691e586fdfb30500a31f2 "ycb-025-mug visual"
check "$ZK0_HOME/assets/ycb-025-mug/params.json" \
      981589df76b70441a0f66a5f4a6efad8 "ycb-025-mug measured params"

# The shared checkpoint, pinned by revision + sha256 (manifest).
CKPT_DIR="$ZK0_HOME/checkpoint/ivelin--zk0-smolvla-fl"
"$NODE_DIR/runtime/venv/bin/python" - "$CKPT_DIR" <<'PYEOF'
import hashlib, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
out = sys.argv[1]
snapshot_download("ivelin/zk0-smolvla-fl",
                  revision="d1c3198fbdb23ea24c9f6751992986af328ce4eb",
                  local_dir=out, token=os.environ.get("HF_TOKEN"))
sha = hashlib.sha256((Path(out) / "model.safetensors").read_bytes()).hexdigest()
want = "693dfdeb68e322ac194d28923ccb9e971e23b38c07cec743705a3b09fbcc7ec4"
if sha != want:
    raise SystemExit(f"bootstrap: checkpoint sha256 mismatch: {sha} != {want}")
print(f"bootstrap: checkpoint ivelin/zk0-smolvla-fl @ d1c3198f sha256 {sha[:16]}… OK")
PYEOF
