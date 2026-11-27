#!/usr/bin/env python3
"""CI self-check: fail on new unit-test skips and on a neutered CI workflow.

Compares tests/unit against the PR/push base (origin/main by default).
The one existing skip is listed in ci/allowed_skips.txt.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "tests" / "unit"
CI_DIR = ROOT / "ci"
WORKFLOWS = ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOWS / "ci.yml"
ALLOWLIST = CI_DIR / "allowed_skips.txt"

# pytest.skip / unittest.skip, skip/skipif/xfail markers, pytest.xfail.
SKIP_RE = re.compile(
    r"""(?x)
    pytest\.skip(?:if)?\s*\(
    | pytest\.xfail\s*\(
    | unittest\.skip(?:If|Unless)?\s*\(
    | @pytest\.mark\.(skip(?:if)?|xfail)\b
    | @unittest\.skip(?:If|Unless)?\b
    """
)

# Patterns that hide a failing pytest / self-check exit status.
MASK_RE = re.compile(
    r"""(?x)
    \|\|\s*true\b
    | continue-on-error\s*:
    | set\s+\+e\b
    """
)

SELF_CHECK_SCAN_SKIP = {"self_check.py"}


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def base_ref() -> str:
    sha = os.environ.get("GITHUB_BASE_SHA", "").strip()
    if sha:
        return sha
    git("fetch", "--no-tags", "--depth", "1", "origin", "main")
    return "origin/main"


def load_allowlist() -> set[str]:
    allowed: set[str] = set()
    if not ALLOWLIST.is_file():
        return allowed
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            allowed.add(line)
    return allowed


def _def_or_class_name(stripped: str) -> str | None:
    if stripped.startswith("async def "):
        return stripped.split("async def ", 1)[1].split("(", 1)[0].strip() or None
    if stripped.startswith("def "):
        return stripped.split("def ", 1)[1].split("(", 1)[0].strip() or None
    if stripped.startswith("class "):
        return stripped.split("class ", 1)[1].split("(", 1)[0].split(":", 1)[0].strip() or None
    return None


def _name_for_skip(lines: list[str], lineno: int, current_def: str) -> str:
    """Decorators apply to the next def/class; in-body skip() uses current_def."""
    for raw in lines[lineno - 1 : lineno + 20]:
        stripped = raw.lstrip()
        if stripped.startswith("@") or stripped.startswith("#") or not stripped:
            continue
        found = _def_or_class_name(stripped)
        if found:
            return found
        break
    return current_def


def skip_hits(text: str, rel_path: str) -> set[tuple[int, str, str]]:
    """Return (lineno, rel_path, function_or_class) for skip/xfail hits."""
    hits: set[tuple[int, str, str]] = set()
    if "live" in Path(rel_path).parts:
        return hits
    lines = text.splitlines()
    current_def = "<module>"
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        found = _def_or_class_name(stripped)
        if found:
            current_def = found
        if SKIP_RE.search(line):
            hits.add((i, rel_path, _name_for_skip(lines, i, current_def)))
    return hits


def file_skips(path: Path) -> set[tuple[int, str, str]]:
    rel = path.relative_to(ROOT).as_posix()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"self-check: cannot read {rel}: {exc}", file=sys.stderr)
        return set()
    return skip_hits(text, rel)


def current_unit_skips() -> set[tuple[int, str, str]]:
    hits: set[tuple[int, str, str]] = set()
    for path in UNIT.rglob("*.py"):
        hits |= file_skips(path)
    return hits


def base_unit_skips(ref: str) -> set[tuple[int, str, str]]:
    listing = git("ls-tree", "-r", "--name-only", ref, "tests/unit")
    hits: set[tuple[int, str, str]] = set()
    for rel in listing.splitlines():
        rel = rel.strip()
        if not rel.endswith(".py"):
            continue
        text = git("show", f"{ref}:{rel}")
        hits |= skip_hits(text, rel)
    return hits


def key(hit: tuple[int, str, str]) -> str:
    _lineno, rel, name = hit
    return f"{rel}::{name}"


def check_new_skips() -> list[str]:
    errors: list[str] = []
    ref = base_ref()
    allowed = load_allowlist()
    current = current_unit_skips()
    base = base_unit_skips(ref)
    base_keys = {key(h) for h in base}
    for hit in sorted(current, key=lambda h: (h[1], h[0])):
        k = key(hit)
        if k in allowed:
            continue
        if k in base_keys:
            continue
        lineno, rel, name = hit
        errors.append(
            f"new skip/xfail in non-live unit test {rel}:{lineno} ({name}) vs {ref}"
        )
    return errors


def uncommented_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        if raw.strip().startswith("#"):
            continue
        out.append(raw)
    return out


def check_ci_not_neutered() -> list[str]:
    errors: list[str] = []
    if not CI_WORKFLOW.is_file():
        return ["missing .github/workflows/ci.yml"]
    body = "\n".join(uncommented_lines(CI_WORKFLOW.read_text(encoding="utf-8")))
    if not re.search(r"\bpytest\b", body):
        errors.append("ci.yml no longer runs pytest (pytest step removed or commented out)")
    if "ci/self_check.py" not in body:
        errors.append("ci.yml does not invoke ci/self_check.py")

    scan_suffixes = {".yml", ".yaml", ".sh", ".txt", ".py"}
    scan_files: list[Path] = [CI_WORKFLOW]
    if CI_DIR.is_dir():
        for path in CI_DIR.rglob("*"):
            if (
                path.is_file()
                and path.suffix in scan_suffixes
                and path.name not in SELF_CHECK_SCAN_SKIP
                and "__pycache__" not in path.parts
            ):
                scan_files.append(path)

    seen: set[Path] = set()
    for path in scan_files:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        rel = path.relative_to(ROOT).as_posix()
        executable = "\n".join(uncommented_lines(path.read_text(encoding="utf-8")))
        if MASK_RE.search(executable):
            errors.append(
                f"neutered CI: exit status masked in {rel} "
                "(|| true, continue-on-error, or set +e)"
            )
        if path == CI_WORKFLOW and re.search(r"exit\s+0\b", executable):
            pytest_pos = executable.find("pytest")
            exit_pos = executable.find("exit 0")
            if exit_pos != -1 and (pytest_pos == -1 or exit_pos < pytest_pos):
                errors.append(f"neutered CI: tests short-circuited with exit 0 in {rel}")
    return errors


def main() -> int:
    errors = check_new_skips() + check_ci_not_neutered()
    if errors:
        print("CI self-check FAILED:")
        for item in errors:
            print(f"  - {item}")
        return 1
    print("CI self-check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
