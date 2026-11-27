#!/usr/bin/env python3
"""Post-merge CI-guard audit for .github/ and ci/ on main."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ZERO_SHA = "0" * 40
LABEL = "ci-guard-escape"
OWNER_LOGIN = "ivelin"
RETRY_SECONDS = 20


def guarded(path: str) -> bool:
    return (
        path == ".github"
        or path.startswith(".github/")
        or path == "ci"
        or path.startswith("ci/")
        or path == "docker/Dockerfile.ci"
        or path == "pyproject.toml"
    )


def flatten_slurp(data):
    """Normalize `gh api --paginate --slurp` JSON (array of pages)."""
    if not isinstance(data, list):
        return data
    if data and isinstance(data[0], list):
        return [item for page in data for item in page]
    return data


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("true", "1", "yes")


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


def commit_pulls(gh: Gh, sha: str) -> list[dict]:
    data = gh.api(f"repos/{gh.repo}/commits/{sha}/pulls", paginate=True)
    if isinstance(data, dict):
        return [data]
    return list(data or [])


def pr_reviews(gh: Gh, number: int) -> list[dict]:
    data = gh.api(f"repos/{gh.repo}/pulls/{number}/reviews", paginate=True)
    if isinstance(data, dict):
        return [data]
    return list(data or [])


def merged_pr_for_commit(gh: Gh, sha: str) -> dict | None:
    for pr in commit_pulls(gh, sha):
        if pr.get("merged_at") and pr.get("merge_commit_sha") == sha:
            return pr
    return None


def ivelin_approved_at_head(reviews: list[dict], head_sha: str) -> bool:
    for review in reviews:
        user = review.get("user") or {}
        if (
            review.get("state") == "APPROVED"
            and user.get("login") == OWNER_LOGIN
            and review.get("commit_id") == head_sha
        ):
            return True
    return False


def ensure_label(gh: Gh) -> bool:
    gh.run(
        [
            "gh",
            "label",
            "create",
            LABEL,
            "--force",
            "--description",
            "CI guard evidence missing",
            "--color",
            "B60205",
        ],
        check=False,
    )
    listed = gh.run(
        ["gh", "api", f"repos/{gh.repo}/labels/{LABEL}"],
        check=False,
    )
    return listed.returncode == 0


def open_escape(gh: Gh, sha: str, missing: str, pr_url: str = "none (direct push)") -> None:
    labeled = ensure_label(gh)
    title = f"ci-guard-escape: {sha[:7]}"
    body = f"SHA: {sha}\nPR: {pr_url}\nMissing: {missing}\n"
    args = ["gh", "issue", "create", "--title", title, "--body", body]
    if labeled:
        args += ["--label", LABEL]
    gh.run(args)
    print(f"ci-guard-escape: {missing}")


def find_merged_pr(gh: Gh, sha: str, sleep=time.sleep, retry_seconds: int = RETRY_SECONDS):
    pr = merged_pr_for_commit(gh, sha)
    if pr:
        return pr
    sleep(retry_seconds)
    return merged_pr_for_commit(gh, sha)


def audit(
    repo: str,
    sha: str,
    before: str = "",
    forced: bool = False,
    event_name: str = "push",
    gh: Gh | None = None,
    sleep=time.sleep,
    retry_seconds: int = RETRY_SECONDS,
) -> int:
    gh = gh or Gh(repo)
    if event_name != "workflow_dispatch":
        if forced:
            open_escape(gh, sha, "forced push to main")
            return 1
        if not before or before == ZERO_SHA:
            open_escape(gh, sha, "push before SHA missing or all zeros")
            return 1
        raw_commits, total = compare_commits(gh, before, sha)
        if total > len(raw_commits):
            open_escape(
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
            open_escape(gh, commit_sha, "empty file list for commit")
            return 1
        if not any(guarded(name) for name in files):
            continue
        pr = find_merged_pr(gh, commit_sha, sleep=sleep, retry_seconds=retry_seconds)
        if not pr:
            open_escape(
                gh,
                commit_sha,
                "no merged PR with merge_commit_sha matching this commit",
            )
            return 1
        head = (pr.get("head") or {}).get("sha") or ""
        reviews = pr_reviews(gh, int(pr["number"]))
        if not ivelin_approved_at_head(reviews, head):
            open_escape(
                gh,
                commit_sha,
                "missing ivelin APPROVED review at PR head SHA",
                pr.get("html_url") or "",
            )
            return 1
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
