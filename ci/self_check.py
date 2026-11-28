#!/usr/bin/env python3
"""CI self-check: new unit-test skips and a neutered CI workflow fail the job.

The workflow runs the *base* copy of this file (and the base allowlist).
This tree is what we inspect. ZK0_ROOT is the checkout; ZK0_ALLOWLIST is
the base allowlist when the job materializes it via git show.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ZERO_SHA = "0" * 40


class GitError(RuntimeError):
    """A git command failed; the self-check must not continue on empty output."""


def repo_root() -> Path:
    env = os.environ.get("ZK0_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip() or f"exit {result.returncode}"
        raise GitError(f"git {' '.join(args)}: {err}")
    return result.stdout


def _nonzero_sha(value: str) -> str:
    sha = value.strip()
    if sha and sha != ZERO_SHA:
        return sha
    return ""


def _event_name() -> str:
    return os.environ.get("GITHUB_EVENT_NAME", "").strip()


def _github_ref_is_main() -> bool:
    """True only for a push/checkout of main. Match ci.yml: GITHUB_REF == refs/heads/main."""
    return os.environ.get("GITHUB_REF", "").strip() == "refs/heads/main"


def _head_sha() -> str:
    sha = _nonzero_sha(os.environ.get("GITHUB_SHA", ""))
    if sha:
        return sha
    return git("rev-parse", "HEAD").strip()


def _merge_base_origin_main() -> str:
    mb = git("merge-base", "origin/main", "HEAD").strip()
    if mb:
        return mb
    raise GitError("origin/main merge-base is empty")


def _refuse_self_diff(sha: str) -> str:
    """Never compare HEAD to itself: that empty range silently passes."""
    head = _head_sha()
    if sha and head and sha == head:
        raise GitError(
            f"base SHA {sha} equals HEAD; refusing empty self-diff "
            "(set GITHUB_BEFORE on push to main, or a PR base SHA)"
        )
    return sha


def base_ref() -> str:
    """Resolve the git range base for skip/CLI/neuter diffs.

    pull_request: GITHUB_BASE_SHA (PR base.sha), else merge-base with origin/main.
    push to main with a non-zero github.event.before: that before SHA.
    all-zeros before or unknown base: merge-base with origin/main, never HEAD.
    """
    event = _event_name()
    pr_base = _nonzero_sha(os.environ.get("GITHUB_BASE_SHA", ""))
    before = _nonzero_sha(os.environ.get("GITHUB_BEFORE", ""))

    if event == "pull_request":
        if pr_base:
            return _refuse_self_diff(pr_base)
        return _refuse_self_diff(_merge_base_origin_main())

    if event == "push" and _github_ref_is_main() and before:
        return _refuse_self_diff(before)

    if event == "push" and pr_base:
        # Workflow may export a resolved SHA; still refuse a no-op self-diff.
        return _refuse_self_diff(pr_base)

    return _refuse_self_diff(_merge_base_origin_main())


def allowlist_path() -> Path:
    env = os.environ.get("ZK0_ALLOWLIST", "").strip()
    if env:
        return Path(env)
    return repo_root() / "ci" / "allowed_skips.txt"


def load_allowlist() -> set[str]:
    path = allowlist_path()
    if not path.is_file():
        return set()
    allowed: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
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
    for raw in lines[lineno - 1 : lineno + 20]:
        stripped = raw.lstrip()
        if stripped.startswith("@") or stripped.startswith("#") or not stripped:
            continue
        found = _def_or_class_name(stripped)
        if found:
            return found
        break
    return current_def


def _parse_from_import(stripped: str) -> dict[str, str]:
    """Map local alias -> skip kind for `from pytest import skip as s` etc."""
    aliases: dict[str, str] = {}
    if not stripped.startswith("from pytest import ") and not stripped.startswith(
        "from unittest import "
    ):
        return aliases
    rest = stripped.split("import ", 1)[1]
    rest = rest.split("#", 1)[0]
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        if " as " in part:
            orig, alias = [p.strip() for p in part.split(" as ", 1)]
        else:
            orig, alias = part, part
        kind_map = {
            "skip": "skip",
            "skipif": "skipif",
            "xfail": "xfail",
            "importorskip": "importorskip",
            "SkipTest": "SkipTest",
            "mark": "mark",
        }
        if orig in kind_map:
            aliases[alias] = kind_map[orig]
    return aliases


SKIP_CALL_RE = re.compile(
    r"""(?x)
    pytest\.skip(?:if)?\s*\(
    | pytest\.xfail\s*\(
    | pytest\.importorskip\s*\(
    | unittest\.skip(?:If|Unless)?\s*\(
    | pytest\.mark\.(skip(?:if)?|xfail)\b
    | @pytest\.mark\.(skip(?:if)?|xfail)\b
    | @unittest\.skip(?:If|Unless)?\b
    | unittest\.SkipTest\b
    | pytestmark\s*=
    | collect_ignore(?:_glob)?\s*=
    | pytest_ignore_collect\b
    | pytest_collection_modifyitems\b
    """
)


def _skip_kind_and_cond(line: str, aliases: dict[str, str]) -> tuple[str, str] | None:
    stripped = line.split("#", 1)[0]
    if SKIP_CALL_RE.search(stripped):
        if "importorskip" in stripped:
            return "importorskip", _call_cond(stripped, "importorskip")
        if "xfail" in stripped:
            return "xfail", _call_cond(stripped, "xfail")
        if "skipif" in stripped:
            return "skipif", _call_cond(stripped, "skipif")
        if "SkipTest" in stripped or "raise " in stripped and "SkipTest" in stripped:
            return "SkipTest", stripped.strip()
        if "pytestmark" in stripped:
            return "pytestmark", stripped.strip()
        if "collect_ignore" in stripped or "pytest_ignore_collect" in stripped:
            return "collect_ignore", stripped.strip()
        if "pytest_collection_modifyitems" in stripped:
            return "collect_hook", stripped.strip()
        return "skip", stripped.strip()
    if re.search(r"\braise\s+.*SkipTest\b", stripped):
        return "SkipTest", stripped.strip()
    for alias, kind in aliases.items():
        if kind == "mark":
            if re.search(rf"\b{re.escape(alias)}\.(skip(?:if)?|xfail)\b", stripped):
                sub = "skipif" if "skipif" in stripped else ("xfail" if "xfail" in stripped else "skip")
                return sub, _call_cond(stripped, sub)
            continue
        if re.search(rf"\b{re.escape(alias)}\s*\(", stripped):
            return kind, _call_cond(stripped, alias)
    return None


def _call_cond(stripped: str, name: str) -> str:
    idx = stripped.find(name)
    if idx < 0:
        return stripped.strip()
    rest = stripped[idx + len(name) :].lstrip()
    if rest.startswith("("):
        depth = 0
        for i, ch in enumerate(rest):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    inner = rest[1:i].strip()
                    inner = inner.split(",", 1)[0].strip()
                    return re.sub(r"\s+", " ", inner)
        return rest.strip()
    return stripped.strip()


def skip_hits(text: str, rel_path: str) -> set[tuple[int, str, str, str, str]]:
    """(lineno, rel, name, kind, cond) for skip/xfail/importorskip/hooks."""
    hits: set[tuple[int, str, str, str, str]] = set()
    if "live" in Path(rel_path).parts:
        return hits
    lines = text.splitlines()
    current_def = "<module>"
    aliases: dict[str, str] = {}
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        aliases.update(_parse_from_import(stripped.split("#", 1)[0].strip()))
        found = _def_or_class_name(stripped)
        if found:
            current_def = found
        parsed = _skip_kind_and_cond(line, aliases)
        if parsed:
            kind, cond = parsed
            name = _name_for_skip(lines, i, current_def)
            hits.add((i, rel_path, name, kind, cond))
    return hits


def skip_key(hit: tuple[int, str, str, str, str]) -> str:
    lineno, rel, name, kind, cond = hit
    return f"{rel}:{lineno}::{name}::{kind}:{cond}"


def skip_fn_key(hit: tuple[int, str, str, str, str]) -> str:
    _lineno, rel, name, kind, cond = hit
    return f"{rel}::{name}::{kind}:{cond}"


def file_skips(path: Path, root: Path) -> set[tuple[int, str, str, str, str]]:
    rel = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8")
    return skip_hits(text, rel)


def current_unit_skips(root: Path) -> set[tuple[int, str, str, str, str]]:
    hits: set[tuple[int, str, str, str, str]] = set()
    tests = root / "tests"
    if not tests.is_dir():
        return hits
    for path in tests.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        under_unit = rel.startswith("tests/unit/")
        if under_unit or Path(rel).name == "conftest.py" or "collect_ignore" in text:
            hits |= skip_hits(text, rel)
    return hits


def base_unit_skips(ref: str) -> set[tuple[int, str, str, str, str]]:
    listing = git("ls-tree", "-r", "--name-only", ref, "tests")
    hits: set[tuple[int, str, str, str, str]] = set()
    for rel in listing.splitlines():
        rel = rel.strip()
        if not rel.endswith(".py"):
            continue
        under_unit = rel.startswith("tests/unit/")
        is_conftest = Path(rel).name == "conftest.py"
        if not under_unit and not is_conftest:
            text_probe = git("show", f"{ref}:{rel}")
            if "collect_ignore" not in text_probe:
                continue
        text = git("show", f"{ref}:{rel}")
        hits |= skip_hits(text, rel)
    return hits


def count_tests_in_text(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("def test_") or stripped.startswith("async def test_"):
            count += 1
    return count


def current_test_files(root: Path) -> dict[str, int]:
    unit = root / "tests" / "unit"
    out: dict[str, int] = {}
    if not unit.is_dir():
        return out
    for path in unit.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        out[rel] = count_tests_in_text(path.read_text(encoding="utf-8"))
    return out


def base_test_files(ref: str) -> dict[str, int]:
    listing = git("ls-tree", "-r", "--name-only", ref, "tests/unit")
    out: dict[str, int] = {}
    for rel in listing.splitlines():
        rel = rel.strip()
        if not rel.endswith(".py"):
            continue
        out[rel] = count_tests_in_text(git("show", f"{ref}:{rel}"))
    return out


def pytest_cli_from_workflow(text: str) -> str:
    cmds = []
    for raw in uncommented_lines(text):
        if re.search(r"\bpytest\b", raw):
            cmds.append(raw.strip())
    return "\n".join(cmds)


def addopts_from_pyproject(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_addopts = False
    for line in lines:
        if line.strip().startswith("addopts"):
            in_addopts = True
        if in_addopts:
            out.append(line.rstrip())
            if in_addopts and line.strip() == "]":
                break
    return "\n".join(out)


def extract_pytest_filters(cli: str) -> tuple[str, str, str]:
    k = ""
    m = ""
    paths = ""
    km = re.search(r"(?:^|\s)-k\s+(\S+)", cli)
    mm = re.search(r"(?:^|\s)-m\s+(\S+)", cli)
    if km:
        k = km.group(1)
    if mm:
        m = mm.group(1)
    path_bits = re.findall(r"(?:^|\s)(tests/\S+)", cli)
    paths = " ".join(path_bits)
    return k, m, paths


def check_new_skips(root: Path | None = None) -> list[str]:
    errors: list[str] = []
    root = root or repo_root()
    ref = base_ref()
    allowed = load_allowlist()
    current = current_unit_skips(root)
    base = base_unit_skips(ref)
    base_keys = {skip_key(h) for h in base} | {skip_fn_key(h) for h in base}
    for hit in sorted(current, key=lambda h: (h[1], h[0])):
        k = skip_key(hit)
        fnk = skip_fn_key(hit)
        if k in allowed or fnk in allowed:
            continue
        if k in base_keys or fnk in base_keys:
            continue
        lineno, rel, name, kind, cond = hit
        errors.append(
            f"new {kind} in {rel}:{lineno} ({name}) cond={cond!r} vs {ref}"
        )
    return errors


def check_deleted_tests_and_count(root: Path | None = None) -> list[str]:
    errors: list[str] = []
    root = root or repo_root()
    ref = base_ref()
    current = current_test_files(root)
    base = base_test_files(ref)
    for rel in sorted(set(base) - set(current)):
        errors.append(f"deleted unit test file {rel} vs {ref}")
    base_n = sum(base.values())
    cur_n = sum(current.values())
    if cur_n < base_n:
        errors.append(
            f"collected test count dropped from {base_n} to {cur_n} vs {ref}"
        )
    return errors


def check_pytest_cli_and_addopts(root: Path | None = None) -> list[str]:
    errors: list[str] = []
    root = root or repo_root()
    ref = base_ref()
    head_wf = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    try:
        base_wf = git("show", f"{ref}:.github/workflows/ci.yml")
    except GitError as exc:
        errors.append(str(exc))
        return errors
    head_cli = pytest_cli_from_workflow(head_wf)
    base_cli = pytest_cli_from_workflow(base_wf)
    hk, hm, hp = extract_pytest_filters(head_cli)
    bk, bm, bp = extract_pytest_filters(base_cli)
    if hk and hk != bk:
        errors.append(f"pytest -k changed ({bk!r} -> {hk!r})")
    if hm and hm != bm:
        errors.append(f"pytest -m changed ({bm!r} -> {hm!r})")
    if hp != bp:
        errors.append(f"pytest test paths changed ({bp!r} -> {hp!r})")
    head_py = (root / "pyproject.toml").read_text(encoding="utf-8")
    base_py = git("show", f"{ref}:pyproject.toml")
    if addopts_from_pyproject(head_py) != addopts_from_pyproject(base_py):
        errors.append("pyproject.toml pytest addopts changed vs base")
    return errors


def uncommented_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        if raw.strip().startswith("#"):
            continue
        out.append(raw)
    return out


def mask_re() -> re.Pattern[str]:
    """Build the neuter regex without writing the forbidden tokens as literals."""
    # chr(124) is a pipe; joining as regex alternatives must wrap each part.
    p2 = re.escape(chr(124) * 2)
    parts = [
        p2 + r"\s*true\b",
        p2 + r"\s*exit\s+0\b",
        p2 + r"\s*" + re.escape(":"),
        r"continue-on-error\s*:",
        r"set\s+\+e\b",
        r"if\s+[Ff]alse\b",
        r"if:\s*\$\{\{\s*false\s*\}\}",
        r"if:\s*false\b",
        r";\s*true\b",
    ]
    return re.compile("|".join(f"(?:{p})" for p in parts))


MASK_SCAN_SKIP_NAMES = {"test_self_check.py"}
MASK_SCAN_SKIP_SUFFIX_PARTS = ("constraints.txt", "constraints-lerobot.txt")


def _self_check_job_block(text: str) -> str | None:
    m = re.search(r"(?m)^  self-check:\n", text)
    if not m:
        return None
    start = m.start()
    n = re.search(r"(?m)^  [A-Za-z0-9_-]+:\n", text[m.end() :])
    end = m.end() + n.start() if n else len(text)
    return text[start:end]


def check_ci_not_neutered(root: Path | None = None) -> list[str]:
    errors: list[str] = []
    root = root or repo_root()
    workflow = root / ".github" / "workflows" / "ci.yml"
    if not workflow.is_file():
        return ["missing .github/workflows/ci.yml"]
    raw = workflow.read_text(encoding="utf-8")
    body = "\n".join(uncommented_lines(raw))
    if not re.search(r"\bpytest\b", body):
        errors.append("ci.yml no longer runs pytest (pytest step removed or commented out)")
    job = _self_check_job_block(raw)
    if job is None:
        errors.append("ci.yml self-check job removed")
    else:
        job_exec = "\n".join(uncommented_lines(job))
        if re.search(r"if:\s*(\$\{\{\s*)?[Ff]alse", job_exec):
            errors.append("ci.yml self-check job is disabled")
        if "self_check.py" not in job_exec:
            errors.append("ci.yml self-check job does not invoke self_check.py")
    if re.search(r"(?m)^  test:", raw) and not re.search(r"needs:\s*self-check", raw):
        errors.append("ci.yml dropped needs: self-check")
    if (root / ".zk0-base-ci").exists():
        errors.append("poison path .zk0-base-ci present")

    scan_suffixes = {".yml", ".yaml", ".sh", ".txt", ".py"}
    scan_files = [workflow, root / "ci" / "self_check.py", root / "docker" / "Dockerfile.ci"]
    ci_dir = root / "ci"
    if ci_dir.is_dir():
        for path in ci_dir.rglob("*"):
            scan_files.append(path)

    seen: set[Path] = set()
    rx = mask_re()
    for path in scan_files:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if path.suffix not in scan_suffixes and path.name != "Dockerfile.ci":
            continue
        if path.name in MASK_SCAN_SKIP_NAMES:
            continue
        if any(path.name.endswith(part) for part in MASK_SCAN_SKIP_SUFFIX_PARTS):
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = str(path)
        executable = "\n".join(uncommented_lines(path.read_text(encoding="utf-8")))
        if rx.search(executable):
            errors.append(f"neutered CI: exit status masked in {rel}")
        if path == workflow:
            pytest_pos = executable.find("pytest")
            for m in re.finditer(r"exit\s+0\b", executable):
                if pytest_pos == -1 or m.start() < pytest_pos:
                    errors.append(f"neutered CI: tests short-circuited with exit 0 in {rel}")
                    break
    head_checker = root / "ci" / "self_check.py"
    if head_checker.is_file():
        src = head_checker.read_text(encoding="utf-8")
        if "check_new_skips" not in src or "check_ci_not_neutered" not in src:
            errors.append("ci/self_check.py gutted (missing check_new_skips / check_ci_not_neutered)")
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            errors.append(f"ci/self_check.py is not valid Python: {exc}")
        else:
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "main":
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                        errors.append("ci/self_check.py main() is a stub return")
    return errors


def ci_review_notices() -> list[str]:  # noqa: D401
    """Non-failing flags so a reviewer must inspect ci/ and ci.yml diffs."""
    try:
        ref = base_ref()
        names = git(
            "diff", "--name-only", ref, "--", "ci/", ".github/workflows/ci.yml"
        )
    except GitError:
        return []
    notices = []
    for rel in names.splitlines():
        rel = rel.strip()
        if rel:
            notices.append(f"REVIEW: {rel} changed vs {ref}; reviewer must inspect")
    return notices


def run_all(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    return (
        check_new_skips(root)
        + check_deleted_tests_and_count(root)
        + check_pytest_cli_and_addopts(root)
        + check_ci_not_neutered(root)
    )


def main() -> int:
    try:
        notices = ci_review_notices()
        errors = run_all()
    except GitError as exc:
        print(f"CI self-check FAILED:\n  - git error: {exc}")
        return 1
    for item in notices:
        print(item)
    if errors:
        print("CI self-check FAILED:")
        for item in errors:
            print(f"  - {item}")
        return 1
    print("CI self-check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
