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
    def __init__(
        self,
        responses: dict[str, Any],
        label_ok: bool = True,
        api_errors: dict[str, int] | None = None,
    ) -> None:
        super().__init__(REPO, run=self._run)
        self.responses = responses
        self.issues: list[list[str]] = []
        self.api_paths: list[str] = []
        self.label_ok = label_ok
        self.api_errors = api_errors or {}

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
            self.api_paths.append(path)
            if path in self.api_errors:
                return FakeProc(returncode=self.api_errors[path])
            payload = self.responses.get(path)
            if payload is None:
                return FakeProc(stdout="[]" if paginate else "{}", returncode=0)
            if paginate:
                slurp_pages = (
                    isinstance(payload, list)
                    and payload
                    and isinstance(payload[0], list)
                )
                if slurp_pages:
                    return FakeProc(stdout=json.dumps(payload))
                return FakeProc(stdout=json.dumps([payload]))
            return FakeProc(stdout=json.dumps(payload))
        raise AssertionError(f"unexpected command: {args}")


def compare(*shas: str, total_commits: int | None = None) -> dict:
    commits = [{"sha": s} for s in shas]
    return {
        "commits": commits,
        "total_commits": len(commits) if total_commits is None else total_commits,
    }


def commit_payload(*files: str, parents: int = 1) -> dict:
    return {
        "files": [{"filename": f} for f in files],
        "parents": [{"sha": f"p{i}"} for i in range(parents)],
    }


def issue_title(args: list[str]) -> str:
    return args[args.index("--title") + 1]


def issue_assignee(args: list[str]) -> str:
    return args[args.index("--assignee") + 1]


def issue_body(args: list[str]) -> str:
    return args[args.index("--body") + 1]


class GuardAuditTests(unittest.TestCase):
    def _audit(self, gh: FakeGh, **kwargs) -> int:
        kwargs.setdefault("sha", MERGE)
        kwargs.setdefault("before", PARENT)
        kwargs.setdefault("forced", False)
        kwargs.setdefault("event_name", "push")
        return ga.audit(REPO, gh=gh, **kwargs)

    def test_every_guarded_change_opens_exactly_one_record(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{SECOND}": compare(FIRST, SECOND),
                f"repos/{REPO}/commits/{FIRST}": commit_payload("ci/guard_audit.py"),
                f"repos/{REPO}/commits/{SECOND}": commit_payload(".github/CODEOWNERS"),
            }
        )
        code = self._audit(gh, sha=SECOND)
        self.assertEqual(code, 0)
        self.assertEqual(len(gh.issues), 1)
        self.assertEqual(issue_title(gh.issues[0]), ga.record_title(SECOND))
        self.assertEqual(issue_assignee(gh.issues[0]), "ivelin")
        body = issue_body(gh.issues[0])
        self.assertIn("ci/guard_audit.py", body)
        self.assertIn(".github/CODEOWNERS", body)
        self.assertIn(f"Range: {PARENT}..{SECOND}", body)
        self.assertTrue(
            all("/reviews" not in path for path in gh.api_paths), gh.api_paths
        )

    def test_unguarded_push_opens_none(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload("README.md"),
            }
        )
        self.assertEqual(self._audit(gh), 0)
        self.assertFalse(gh.issues)

    def test_open_issues_list_failure_exits_nonzero_without_record(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload(".github/CODEOWNERS"),
            },
            api_errors={f"repos/{REPO}/{ga.ISSUES_OPEN_PATH}": 1},
        )
        with self.assertRaises(SystemExit) as ctx:
            self._audit(gh)
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertFalse(gh.issues)
        self.assertTrue(
            any(ga.ISSUES_OPEN_PATH in path for path in gh.api_paths), gh.api_paths
        )

    def test_s14_skips_create_when_open_record_on_page_two(self) -> None:
        title = ga.record_title(MERGE)
        pages = [
            [{"title": "unrelated open issue", "state": "open"}],
            [{"title": title, "state": "open"}],
        ]
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload(".github/CODEOWNERS"),
                f"repos/{REPO}/{ga.ISSUES_OPEN_PATH}": pages,
            }
        )
        self.assertEqual(self._audit(gh), 0)
        self.assertFalse(gh.issues)
        self.assertTrue(
            any(ga.ISSUES_OPEN_PATH in path for path in gh.api_paths), gh.api_paths
        )

    def test_s14_skips_create_when_open_record_exists(self) -> None:
        title = ga.record_title(MERGE)
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload(".github/CODEOWNERS"),
                f"repos/{REPO}/{ga.ISSUES_OPEN_PATH}": [
                    {"title": title, "state": "open"}
                ],
            }
        )
        self.assertEqual(self._audit(gh), 0)
        self.assertFalse(gh.issues)
        self.assertTrue(
            any(ga.ISSUES_OPEN_PATH in path for path in gh.api_paths), gh.api_paths
        )
        self.assertTrue(
            all("/reviews" not in path for path in gh.api_paths), gh.api_paths
        )

    def test_s11_truncated_compare_opens_record_multi_page(self) -> None:
        pages = [
            {"commits": [{"sha": FIRST}], "total_commits": 5},
            {"commits": [{"sha": MERGE}], "total_commits": 5},
        ]
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": pages,
                f"repos/{REPO}/commits/{FIRST}": commit_payload("README.md"),
                f"repos/{REPO}/commits/{MERGE}": commit_payload("docs/x.md"),
            }
        )
        self.assertEqual(self._audit(gh), 1)
        self.assertEqual(len(gh.issues), 1)
        self.assertEqual(issue_title(gh.issues[0]), ga.record_title(MERGE))
        self.assertIn("total_commits=5", issue_body(gh.issues[0]))

    def test_empty_file_list_on_non_merge_opens_record(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload(parents=1),
            }
        )
        self.assertEqual(self._audit(gh), 1)
        self.assertEqual(len(gh.issues), 1)
        self.assertEqual(issue_title(gh.issues[0]), ga.record_title(MERGE))

    def test_forced_push_opens_record(self) -> None:
        gh = FakeGh({})
        self.assertEqual(self._audit(gh, forced=True), 1)
        self.assertEqual(len(gh.issues), 1)
        self.assertEqual(issue_title(gh.issues[0]), ga.record_title(MERGE))

    def test_before_all_zeros_opens_record(self) -> None:
        gh = FakeGh({})
        self.assertEqual(self._audit(gh, before=ga.ZERO_SHA), 1)
        self.assertEqual(len(gh.issues), 1)

    def test_record_created_without_label_if_label_missing(self) -> None:
        gh = FakeGh(
            {
                f"repos/{REPO}/compare/{PARENT}...{MERGE}": compare(MERGE),
                f"repos/{REPO}/commits/{MERGE}": commit_payload("ci/x.py"),
            },
            label_ok=False,
        )
        self.assertEqual(self._audit(gh), 0)
        self.assertNotIn("--label", gh.issues[0])

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
        self.assertEqual(len(gh.issues), 1)

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
        self.assertEqual(len(gh.issues), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
