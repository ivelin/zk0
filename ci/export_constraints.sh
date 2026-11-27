#!/usr/bin/env bash
# Regenerate ci/constraints.txt and ci/constraints-lerobot.txt from zk0-ci:bookworm.
# Mirrors .github/workflows/ci.yml:
#   uv pip install --system --constraint ci/constraints.txt -e '.[test]'
# Run from the repo root after: docker build -f docker/Dockerfile.ci -t zk0-ci:bookworm .
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

header_of() {
  python3 -c '
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text(encoding="utf-8")
out = []
for ln in text.splitlines(True):
    if ln.startswith("#") or not ln.strip():
        out.append(ln)
        continue
    break
sys.stdout.write("".join(out).rstrip() + "\n")
' "$1"
}

sort_pins() {
  python3 -c '
import sys
lines = []
for ln in sys.stdin:
    ln = ln.strip()
    if not ln or ln.startswith("#"):
        continue
    if ln.startswith("zk0==") or ln.startswith("-e "):
        continue
    lines.append(ln)
print("\n".join(sorted(lines, key=str.lower)))
'
}

test_header="$(header_of ci/constraints.txt)"
lero_header="$(header_of ci/constraints-lerobot.txt)"

{
  printf '%s\n' "$test_header"
  docker run --rm -v "${PWD}:/workspace" zk0-ci:bookworm bash -c "
    cd /workspace &&
    uv pip install --system --constraint ci/constraints.txt -e '.[test]' &&
    python -m pip freeze
  " | sort_pins
  printf '\n'
} > ci/constraints.txt

{
  printf '%s\n' "$lero_header"
  docker run --rm zk0-ci:bookworm python -m pip freeze | sort_pins
  printf '\n'
} > ci/constraints-lerobot.txt

echo "Wrote ci/constraints.txt and ci/constraints-lerobot.txt"
