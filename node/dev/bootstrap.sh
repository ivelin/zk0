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
