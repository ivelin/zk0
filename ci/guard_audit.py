#!/usr/bin/env python3
"""Post-merge CI-guard audit: one guarded-merge record per main push."""
from __future__ import annotations

import json
import os
import subprocess
import sys

ZERO_SHA = "0" * 40
OWNER_LOGIN = "ivelin"
LABEL = "guarded-merge"
ISSUES_OPEN_PATH = "issues?state=open&per_page=100"


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


def record_title(sha: str) -> str:
    return f"human review needed: guarded-merge record {sha}"


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


def commit_detail(gh: Gh, sha: str) -> dict:
    data = gh.api(f"repos/{gh.repo}/commits/{sha}", paginate=True)
    if isinstance(data, list):
        files: list[dict] = []
        parents: list[dict] = []
        for page in data:
            if isinstance(page, dict):
                files.extend(page.get("files") or [])
                parents = page.get("parents") or parents
        return {"files": files, "parents": parents}
    if isinstance(data, dict):
        return data
    return {}


def commit_files(detail: dict) -> list[str]:
    return [f.get("filename", "") for f in (detail.get("files") or []) if f.get("filename")]


def is_merge_commit(detail: dict) -> bool:
    return len(detail.get("parents") or []) > 1


def commit_pulls(gh: Gh, sha: str) -> list[dict]:
    data = gh.api(f"repos/{gh.repo}/commits/{sha}/pulls", paginate=True)
    if isinstance(data, dict):
        return [data]
    return list(data or [])


def associated_pr(gh: Gh, sha: str) -> dict | None:
    for pr in commit_pulls(gh, sha):
        if pr.get("merged_at") and pr.get("merge_commit_sha") == sha:
            return pr
    pulls = commit_pulls(gh, sha)
    return pulls[0] if pulls else None


def open_issues(gh: Gh) -> list[dict]:
    data = gh.api(f"repos/{gh.repo}/{ISSUES_OPEN_PATH}", paginate=True)
    if isinstance(data, dict):
        return [data]
    return list(data or [])


def has_open_record(gh: Gh, title: str) -> bool:
    for issue in open_issues(gh):
        if issue.get("title") == title and "pull_request" not in issue:
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
            "Guarded-path merge record",
            "--color",
            "0E8A16",
        ],
        check=False,
    )
    listed = gh.run(["gh", "api", f"repos/{gh.repo}/labels/{LABEL}"], check=False)
    return listed.returncode == 0


def record_body(
    sha: str,
    before: str,
    guarded_files: list[str],
    pr: dict | None,
    note: str = "",
) -> str:
    files = "\n".join(f"- {name}" for name in guarded_files) or "- (none listed)"
    pr_link = (pr or {}).get("html_url") or "none"
    extra = f"{note}\n" if note else ""
    return (
        f"Merged SHA: {sha}\n"
        f"Range: {before}..{sha}\n"
        f"Guarded files touched:\n{files}\n"
        f"PR: {pr_link}\n"
        f"{extra}"
    )


def open_record(gh: Gh, sha: str, body: str) -> None:
    title = record_title(sha)
    args = [
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
    if ensure_label(gh):
        args += ["--label", LABEL]
    gh.run(args)
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
    fail_closed = False
    note = ""
    shas: list[str]
    if event_name != "workflow_dispatch":
        if forced:
            fail_closed = True
            note = "forced push to main"
            shas = [sha]
        elif not before or before == ZERO_SHA:
            fail_closed = True
            note = "push before SHA missing or all zeros"
            shas = [sha]
        else:
            raw_commits, total = compare_commits(gh, before, sha)
            if total > len(raw_commits):
                fail_closed = True
                note = (
                    f"compare truncated: got {len(raw_commits)} commits, "
                    f"total_commits={total}"
                )
            shas = [c.get("sha") for c in raw_commits if c.get("sha")]
            if sha not in shas:
                shas.append(sha)
    else:
        shas = [sha]

    guarded_files: list[str] = []
    seen: set[str] = set()
    empty_non_merge = False
    for commit_sha in shas:
        detail = commit_detail(gh, commit_sha)
        files = commit_files(detail)
        if not files and not is_merge_commit(detail):
            empty_non_merge = True
        for name in files:
            if guarded(name) and name not in seen:
                seen.add(name)
                guarded_files.append(name)

    if empty_non_merge:
        fail_closed = True
        note = note or "empty file list on a non-merge commit"

    should_record = bool(guarded_files) or fail_closed
    if should_record:
        title = record_title(sha)
        if has_open_record(gh, title):
            print(f"record exists: {title}")
        else:
            pr = associated_pr(gh, sha)
            body = record_body(sha, before or "(none)", guarded_files, pr, note)
            open_record(gh, sha, body)
        if fail_closed:
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
