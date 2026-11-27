#!/usr/bin/env python3
"""Tests for ci/guard_audit.py. Run: python3 -I ci/test_guard_audit.py"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard_audit as ga  # noqa: E402

REPO = "ivelin/zk0"
MERGE = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
PARENT = "cccccccccccccccccccccccccccccccccccccccc"
FIRST = "1111111111111111111111111111111111111111"
SECOND = "2222222222222222222222222222222222222222"


class FakeProc:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


class FakeGh(ga.Gh):
    def __init__(self, responses: dict[str, Any]) -> None:
        super().__init__(REPO, run=self._run)
        self.responses = responses
        self.issues: list[list[str]] = []

    def _run(self, args: list[str], capture_output: bool = True, text: bool = True):
        if args[:3] == ["gh", "issue", "create"]:
            self.issues.append(args)
            return FakeProc()
        if args[:2] == ["gh", "api"]:
            paginate = "--paginate" in args and "--slurp" in args
            path = args[-1]
            payload = self.responses.get(path)
            if payload is None:
                return FakeProc(stdout="[]" if paginate else "{}", returncode=0)
            if paginate:
                return FakeProc(stdout=json.dumps([payload]))
            return FakeProc(stdout=json.dumps(payload))
        raise AssertionError(f"unexpected command: {args}")


def compare(*shas: str, total_commits: int | None = None) -> dict:
    commits = [{"sha": s} for s in shas]
    return {
        "commits": commits,
        "total_commits": len(commits) if total_commits is None else total_commits,
    }


def commit_payload(*files: str) -> dict:
    return {"files": [{"filename": f} for f in files]}


def issue_title(args: list[str]) -> str:
    return args[args.index("--title") + 1]


def issue_assignee(args: list[str]) -> str:
    return args[args.index("--assignee") + 1]


class GuardAuditTests(unittest.TestCase):
    def _audit(self, gh: FakeGh, **kwargs) -> int:
        kwargs.setdefault("sha", MERGE)
        kwargs.setdefault("before", PARENT)
        kwargs.setdefault("forced", False)
        kwargs.setdefault("event_name", "push")
        return ga.audit(REPO, gh=gh, **kwargs)

    def test_guarded_change_in_first_of_two_commits_opens_issue(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{SECOND}": compare(FIRST, SECOND),
                f"repos/{REPO}/commits/{FIRST}": commit_payload("ci/guard_audit.py"),
                f"repos/{REPO}/commits/{SECOND}": commit_payload("README.md"),
            }
        )
        code = self._audit(gh, sha=SECOND)
        self.assertEqual(code, 0)
        self.assertEqual(len(gh.issues), 1)
        self.assertEqual(issue_title(gh.issues[0]), ga.review_issue_title(FIRST))
        self.assertEqual(issue_assignee(gh.issues[0]), "ivelin")

    def test_unguarded_push_opens_no_issue(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload("README.md", "docs/x.md"),
            }
        )
        self.assertEqual(self._audit(gh), 0)
        self.assertFalse(gh.issues)

    def test_empty_file_list_fails_closed_and_opens_issue(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload(),
            }
        )
        self.assertEqual(self._audit(gh), 1)
        self.assertEqual(len(gh.issues), 1)
        self.assertEqual(issue_title(gh.issues[0]), ga.review_issue_title(MERGE))
        self.assertEqual(issue_assignee(gh.issues[0]), "ivelin")

    def test_compare_truncated_fails_closed_and_opens_issue(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(
                    MERGE, total_commits=3
                ),
            }
        )
        self.assertEqual(self._audit(gh), 1)
        self.assertEqual(len(gh.issues), 1)
        self.assertEqual(issue_title(gh.issues[0]), ga.review_issue_title(MERGE))
        self.assertEqual(issue_assignee(gh.issues[0]), "ivelin")

    def test_before_all_zeros_fails_closed_and_opens_issue(self) -> None:
        gh = FakeGh({})
        code = self._audit(gh, before=ga.ZERO_SHA)
        self.assertEqual(code, 1)
        self.assertEqual(issue_title(gh.issues[0]), ga.review_issue_title(MERGE))
        self.assertEqual(issue_assignee(gh.issues[0]), "ivelin")

    def test_forced_push_fails_closed_and_opens_issue(self) -> None:
        gh = FakeGh({})
        code = self._audit(gh, forced=True)
        self.assertEqual(code, 1)
        self.assertEqual(issue_title(gh.issues[0]), ga.review_issue_title(MERGE))
        self.assertEqual(issue_assignee(gh.issues[0]), "ivelin")

    def test_flatten_slurp_pages(self) -> None:
        self.assertEqual(ga.flatten_slurp([[1], [2, 3]]), [1, 2, 3])
        self.assertEqual(ga.flatten_slurp([{"a": 1}]), [{"a": 1}])


class Ba1ConditionalGuardedPathsTests(unittest.TestCase):
    """Ari BA1: conditional on Ivelin; may be dropped with CODEOWNERS BA1 lines."""

    def test_dockerfile_ci_is_guarded(self) -> None:
        self.assertTrue(ga.guarded("docker/Dockerfile.ci"))
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload("docker/Dockerfile.ci"),
            }
        )
        code = ga.audit(REPO, MERGE, before=PARENT, gh=gh)
        self.assertEqual(code, 0)
        self.assertEqual(issue_title(gh.issues[0]), ga.review_issue_title(MERGE))
        self.assertEqual(issue_assignee(gh.issues[0]), "ivelin")

    def test_pyproject_toml_is_guarded(self) -> None:
        self.assertTrue(ga.guarded("pyproject.toml"))
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload("pyproject.toml"),
            }
        )
        code = ga.audit(REPO, MERGE, before=PARENT, gh=gh)
        self.assertEqual(code, 0)
        self.assertEqual(issue_title(gh.issues[0]), ga.review_issue_title(MERGE))
        self.assertEqual(issue_assignee(gh.issues[0]), "ivelin")


if __name__ == "__main__":
    unittest.main(verbosity=2)
