#!/usr/bin/env python3
"""Tests for ci/guard_audit.py. Run: python3 ci/test_guard_audit.py"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard_audit as ga  # noqa: E402

REPO = "ivelin/zk0"
HEAD = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
MERGE = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
PARENT = "cccccccccccccccccccccccccccccccccccccccc"
OTHER = "dddddddddddddddddddddddddddddddddddddddd"
FIRST = "1111111111111111111111111111111111111111"
SECOND = "2222222222222222222222222222222222222222"


class FakeProc:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


class FakeGh(ga.Gh):
    def __init__(self, responses: dict[str, Any], label_ok: bool = True) -> None:
        super().__init__(REPO, run=self._run)
        self.responses = responses
        self.issues: list[list[str]] = []
        self.label_ok = label_ok
        self.api_calls: list[str] = []

    def _run(self, args: list[str], capture_output: bool = True, text: bool = True):
        if args[:2] == ["gh", "label"] and "create" in args:
            return FakeProc(returncode=0 if self.label_ok else 1)
        if args[:3] == ["gh", "api", f"repos/{REPO}/labels/{ga.LABEL}"]:
            return FakeProc(returncode=0 if self.label_ok else 1)
        if args[:3] == ["gh", "issue", "create"]:
            self.issues.append(args)
            return FakeProc()
        if args[:2] == ["gh", "api"]:
            paginate = "--paginate" in args and "--slurp" in args
            path = args[-1]
            self.api_calls.append(path)
            payload = self.responses.get(path)
            if payload is None:
                return FakeProc(stdout="[]" if paginate else "{}", returncode=0)
            if paginate:
                return FakeProc(stdout=json.dumps([payload]))
            return FakeProc(stdout=json.dumps(payload))
        raise AssertionError(f"unexpected command: {args}")


def merged_pr(head: str = HEAD, merge: str = MERGE, number: int = 71) -> dict:
    return {
        "number": number,
        "merged_at": "2026-11-27T00:00:00Z",
        "merge_commit_sha": merge,
        "html_url": f"https://github.com/{REPO}/pull/{number}",
        "head": {"sha": head},
    }


def review(login: str, commit_id: str, state: str = "APPROVED") -> dict:
    return {"state": state, "user": {"login": login}, "commit_id": commit_id}


def compare(*shas: str) -> dict:
    return {"commits": [{"sha": s} for s in shas]}


def commit_payload(*files: str) -> dict:
    return {"files": [{"filename": f} for f in files]}


class GuardAuditTests(unittest.TestCase):
    def _audit(self, gh: FakeGh, **kwargs) -> int:
        kwargs.setdefault("sleep", lambda _s: None)
        kwargs.setdefault("retry_seconds", 0)
        return ga.audit(REPO, MERGE, before=PARENT, forced=False, event_name="push", gh=gh, **kwargs)

    def test_forged_author_approved_review_escapes(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload(".github/CODEOWNERS"),
                f"repos/{REPO}/commits/{MERGE}/pulls": [merged_pr()],
                f"repos/{REPO}/pulls/71/reviews": [review("not-ivelin", HEAD)],
            }
        )
        self.assertEqual(self._audit(gh), 1)
        self.assertTrue(gh.issues)

    def test_approval_on_stale_commit_id_escapes(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload("ci/self_check.py"),
                f"repos/{REPO}/commits/{MERGE}/pulls": [merged_pr()],
                f"repos/{REPO}/pulls/71/reviews": [review("ivelin", OTHER)],
            }
        )
        self.assertEqual(self._audit(gh), 1)
        self.assertTrue(gh.issues)

    def test_comment_only_fail_pass_pr_escapes(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload(".github/workflows/ci.yml"),
                f"repos/{REPO}/commits/{MERGE}/pulls": [merged_pr()],
                f"repos/{REPO}/pulls/71/reviews": [],
            }
        )
        self.assertEqual(self._audit(gh), 1)
        self.assertTrue(gh.issues)

    def test_two_commit_direct_push_ci_in_first_escapes(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{SECOND}": compare(FIRST, SECOND),
                f"repos/{REPO}/commits/{FIRST}": commit_payload("ci/guard_audit.py"),
                f"repos/{REPO}/commits/{SECOND}": commit_payload("README.md"),
                f"repos/{REPO}/commits/{FIRST}/pulls": [],
                f"repos/{REPO}/commits/{SECOND}/pulls": [],
            }
        )
        code = ga.audit(
            REPO,
            SECOND,
            before=PARENT,
            gh=gh,
            sleep=lambda _s: None,
            retry_seconds=0,
        )
        self.assertEqual(code, 1)
        self.assertTrue(gh.issues)
        self.assertIn(FIRST[:7], " ".join(gh.issues[0]))

    def test_before_all_zeros_escapes(self) -> None:
        gh = FakeGh({})
        code = ga.audit(REPO, MERGE, before=ga.ZERO_SHA, gh=gh, sleep=lambda _s: None)
        self.assertEqual(code, 1)
        self.assertTrue(gh.issues)

    def test_forced_push_escapes(self) -> None:
        gh = FakeGh({})
        code = ga.audit(REPO, MERGE, before=PARENT, forced=True, gh=gh, sleep=lambda _s: None)
        self.assertEqual(code, 1)
        self.assertTrue(gh.issues)

    def test_unmerged_open_pr_associated_escapes(self) -> None:
        open_pr = merged_pr()
        open_pr["merged_at"] = None
        open_pr["merge_commit_sha"] = None
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload(".github/CODEOWNERS"),
                f"repos/{REPO}/commits/{MERGE}/pulls": [open_pr],
                f"repos/{REPO}/pulls/71/reviews": [review("ivelin", HEAD)],
            }
        )
        self.assertEqual(self._audit(gh), 1)
        self.assertTrue(gh.issues)

    def test_ivelin_approved_at_head_on_merged_pr_passes(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload(".github/CODEOWNERS"),
                f"repos/{REPO}/commits/{MERGE}/pulls": [merged_pr()],
                f"repos/{REPO}/pulls/71/reviews": [review("ivelin", HEAD)],
            }
        )
        self.assertEqual(self._audit(gh), 0)
        self.assertFalse(gh.issues)

    def test_push_touching_no_guarded_paths_passes(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload("README.md", "docs/x.md"),
            }
        )
        self.assertEqual(self._audit(gh), 0)
        self.assertFalse(gh.issues)

    def test_flatten_slurp_pages(self) -> None:
        self.assertEqual(ga.flatten_slurp([[1], [2, 3]]), [1, 2, 3])
        self.assertEqual(ga.flatten_slurp([{"a": 1}]), [{"a": 1}])

    def test_issue_created_without_label_if_label_missing(self) -> None:
        gh = FakeGh({}, label_ok=False)
        ga.open_escape(gh, MERGE, "missing")
        self.assertTrue(gh.issues)
        self.assertNotIn("--label", gh.issues[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
