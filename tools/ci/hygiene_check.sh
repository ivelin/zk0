#!/usr/bin/env bash
# hygiene_check.sh — report-only repository hygiene check (DRAFT).
#
# Phase 0 draft of the CI hygiene gate from the reproducibility remediation
# plan. It PRINTS warnings and ALWAYS EXITS 0 — it is not wired as an
# enforced CI gate yet. Enforcement wiring is Phase 2 work.
#
# Checks:
#   1. Workstation-specific absolute paths committed to the repo.
#   2. Tailnet (100.64.0.0/10) and LAN (192.168.0.0/16) address literals.
#   3. Pointers to private systems of record ("Drive SSOT" strings) in
#      public content — public docs must be self-contained.
#   4. Broken relative links in Markdown files.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
warn=0
note() { echo "hygiene: $*"; warn=1; }

echo "== 1. workstation paths =="
grep -rn --exclude-dir=.git -E '/home/(openclaw|[a-z_][a-z0-9_-]*)' . \
  | grep -v 'tools/ci/hygiene_check.sh' \
  | while IFS= read -r line; do note "$line"; done

echo "== 2. tailnet / LAN address literals =="
grep -rn --exclude-dir=.git -E '100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}' . \
  | grep -v 'tools/ci/hygiene_check.sh' \
  | while IFS= read -r line; do note "$line"; done

echo "== 3. private-record pointers =="
grep -rn --exclude-dir=.git 'Drive SSOT' . \
  | grep -v 'tools/ci/hygiene_check.sh' \
  | while IFS= read -r line; do note "$line"; done

echo "== 4. broken relative markdown links =="
python3 - <<'PYEOF'
import os, re, sys
link_re = re.compile(r'\[[^\]]*\]\(([^)]+)\)')
broken = 0
for dirpath, dirnames, filenames in os.walk('.'):
    if '.git' in dirpath.split(os.sep):
        continue
    for fn in filenames:
        if not fn.endswith('.md'):
            continue
        path = os.path.join(dirpath, fn)
        try:
            text = open(path, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        for target in link_re.findall(text):
            target = target.split('#')[0].strip()
            if not target or '://' in target or target.startswith('mailto:'):
                continue
            if not os.path.exists(os.path.normpath(os.path.join(dirpath, target))):
                print(f"hygiene: {path}: broken relative link -> {target}")
                broken += 1
print(f"hygiene: broken relative links found: {broken}")
PYEOF

echo "hygiene: report-only draft — exiting 0 regardless of findings."
exit 0
