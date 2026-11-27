#!/usr/bin/env bash
# Regenerate ci/constraints.txt from the pinned CI image after -e '.[test]'.
# Run from the repo root:
#   docker build -f docker/Dockerfile.ci -t zk0-ci:bookworm .
#   ./ci/export_constraints.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
docker run --rm -v "${PWD}:/workspace" zk0-ci:bookworm bash -c "
  cd /workspace &&
  uv pip install --system -e '.[test]' &&
  python -m pip freeze | grep -v -E '^(zk0==|-e )'
" > ci/constraints.txt
echo "Wrote ci/constraints.txt"
