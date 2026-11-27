#!/usr/bin/env python3
"""Post-merge CI-guard audit: record guarded main-push commits for human review."""
from __future__ import annotations

import json
import os
import subprocess
import sys

ZERO_SHA = "0" * 40
OWNER_LOGIN = "ivelin"


def guarded(path: str) -> bool:
    if (
        path == ".github"
        or path.startswith(".github/")
        or path == "ci"
        or path.startswith("ci/")
    ):
        return True
    # --- Ari BA1 (conditional on Ivelin; may be dropped) ---
    return path == "docker/Dockerfile.ci" or path == "pyproject.toml"


def flatten_slurp(data):
    """Normalize `gh api --paginate --slurp` JSON (array of pages)."""
    if not isinstance(data, list):
        return data
    if data and isinstance(data[0], list):
        return [item for page in data for item in page]
    return data


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("true", "1", "yes")


def review_issue_title(head_sha: str) -> str:
    return f"zk0-ci-guard: human review needed {head_sha}"


class Gh:
    def __init__(self, repo: str, run=None) -> None:
        self.repo = repo
        self._run = run or subprocess.run

    def run(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        proc = self._run(args, capture_output=True, text=True)
        if check and proc.returncode:
            sys.stderr.write(proc.stderr or "")
            raise SystemExit(proc.returncode)
        return proc

    def api(self, path: str, paginate: bool = False):
        cmd = ["gh", "api"]
        if paginate:
            cmd += ["--paginate", "--slurp"]
        cmd.append(path)
        out = (self.run(cmd).stdout or "").strip()
        if not out:
            return [] if paginate else {}
        data = json.loads(out)
        return flatten_slurp(data) if paginate else data


def compare_commits(gh: Gh, before: str, sha: str) -> tuple[list[dict], int]:
    pages = gh.api(f"repos/{gh.repo}/compare/{before}...{sha}", paginate=True)
    if isinstance(pages, dict):
        pages = [pages]
    commits: list[dict] = []
    total = 0
    for page in pages or []:
        if isinstance(page, dict):
            commits.extend(page.get("commits") or [])
            reported = page.get("total_commits")
            if reported is not None:
                total = max(total, int(reported))
    return commits, total


def commit_files(gh: Gh, sha: str) -> list[str]:
    data = gh.api(f"repos/{gh.repo}/commits/{sha}", paginate=True)
    if isinstance(data, dict):
        files = data.get("files") or []
    elif isinstance(data, list):
        files = []
        for page in data:
            if isinstance(page, dict):
                files.extend(page.get("files") or [])
    else:
        files = []
    return [f.get("filename", "") for f in files if f.get("filename")]


def open_review_issue(gh: Gh, head_sha: str, body: str) -> None:
    title = review_issue_title(head_sha)
    gh.run(
        [
            "gh",
            "issue",
            "create",
            "--title",
            title,
            "--assignee",
            OWNER_LOGIN,
            "--body",
            body,
        ]
    )
    print(title)


def audit(
    repo: str,
    sha: str,
    before: str = "",
    forced: bool = False,
    event_name: str = "push",
    gh: Gh | None = None,
) -> int:
    gh = gh or Gh(repo)
    if event_name != "workflow_dispatch":
        if forced:
            open_review_issue(gh, sha, "forced push to main")
            return 1
        if not before or before == ZERO_SHA:
            open_review_issue(gh, sha, "push before SHA missing or all zeros")
            return 1
        raw_commits, total = compare_commits(gh, before, sha)
        if total > len(raw_commits):
            open_review_issue(
                gh,
                sha,
                f"compare truncated: got {len(raw_commits)} commits, total_commits={total}",
            )
            return 1
        shas = [c.get("sha") for c in raw_commits if c.get("sha")]
        if sha not in shas:
            shas.append(sha)
    else:
        shas = [sha]

    for commit_sha in shas:
        files = commit_files(gh, commit_sha)
        if not files:
            open_review_issue(gh, commit_sha, "empty file list for commit")
            return 1
        if not any(guarded(name) for name in files):
            continue
        open_review_issue(
            gh,
            commit_sha,
            f"SHA: {commit_sha}\nGuarded paths landed on main; human review needed.\n",
        )
    print("ci-guard-audit OK")
    return 0


def main() -> int:
    return audit(
        repo=os.environ["REPO"],
        sha=os.environ["SHA"],
        before=os.environ.get("BEFORE") or "",
        forced=truthy(os.environ.get("FORCED")),
        event_name=os.environ.get("EVENT_NAME") or "push",
    )


if __name__ == "__main__":
    sys.exit(main())
