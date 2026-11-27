#!/usr/bin/env python3
"""Harness for ci/self_check.py. Run: python3 ci/test_self_check.py"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import self_check as sc  # noqa: E402


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


MIN_CHECKER = '''def check_new_skips(root=None):
    return []
def check_ci_not_neutered(root=None):
    return []
def main():
    errors = check_new_skips() + check_ci_not_neutered()
    return 1 if errors else 0
'''

MIN_CI_YML = """name: CI
on: [push]
jobs:
  self-check:
    runs-on: ubuntu-latest
    steps:
    - run: python3 ci/self_check.py
  test:
    runs-on: ubuntu-latest
    steps:
    - run: pytest tests/unit
"""


class SkipDetectionTests(unittest.TestCase):
    def test_pytestmark_skip(self) -> None:
        text = "import pytest\npytestmark = pytest.mark.skip\ndef test_a():\n    assert True\n"
        hits = sc.skip_hits(text, "tests/unit/test_x.py")
        kinds = {h[3] for h in hits}
        self.assertIn("pytestmark", kinds)

    def test_importorskip(self) -> None:
        text = "import pytest\npytest.importorskip('missingpkg')\ndef test_a():\n    pass\n"
        hits = sc.skip_hits(text, "tests/unit/test_x.py")
        kinds = {h[3] for h in hits}
        self.assertIn("importorskip", kinds)

    def test_raise_skiptest(self) -> None:
        text = "from unittest import SkipTest\ndef test_a():\n    raise SkipTest('nope')\n"
        hits = sc.skip_hits(text, "tests/unit/test_x.py")
        kinds = {h[3] for h in hits}
        self.assertIn("SkipTest", kinds)

    def test_from_import_skip(self) -> None:
        text = "from pytest import skip\ndef test_a():\n    skip('nope')\n"
        hits = sc.skip_hits(text, "tests/unit/test_x.py")
        kinds = {h[3] for h in hits}
        self.assertIn("skip", kinds)

    def test_undecorated_mark_skipif(self) -> None:
        text = "import pytest\nmarker = pytest.mark.skipif(True, reason='x')\ndef test_a():\n    pass\n"
        hits = sc.skip_hits(text, "tests/unit/test_x.py")
        kinds = {h[3] for h in hits}
        self.assertIn("skipif", kinds)

    def test_collect_ignore(self) -> None:
        text = "collect_ignore = ['test_foo.py']\n"
        hits = sc.skip_hits(text, "tests/conftest.py")
        kinds = {h[3] for h in hits}
        self.assertIn("collect_ignore", kinds)

    def test_skipif_true_key(self) -> None:
        text = (
            "import pytest\nclass T:\n"
            "    @pytest.mark.skipif(True, reason='x')\n"
            "    def test_client_fn_loads_env(self):\n        pass\n"
        )
        hits = sc.skip_hits(text, "tests/unit/test_client_app.py")
        keys = {sc.skip_key(h) for h in hits}
        self.assertTrue(any("skipif:True" in k and "test_client_fn_loads_env" in k for k in keys))

    def test_allowlist_comes_from_env_not_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "ci/allowed_skips.txt", "tests/unit/foo.py:1::test_a::skip:x\n")
            _write(root, "base-allow.txt", "")
            os.environ["ZK0_ALLOWLIST"] = str(root / "base-allow.txt")
            os.environ["ZK0_ROOT"] = str(root)
            try:
                allowed = sc.load_allowlist()
            finally:
                os.environ.pop("ZK0_ALLOWLIST", None)
                os.environ.pop("ZK0_ROOT", None)
            self.assertEqual(allowed, set())


class NeuterDetectionTests(unittest.TestCase):
    def _tree(self, ci_yml: str, checker: str = MIN_CHECKER) -> Path:
        root = Path(tempfile.mkdtemp())
        _write(root, ".github/workflows/ci.yml", ci_yml)
        _write(root, "ci/self_check.py", checker)
        _write(root, "docker/Dockerfile.ci", "FROM python:3.10-slim-bookworm\n")
        return root

    def test_or_exit_0_in_ci_yml(self) -> None:
        yml = MIN_CI_YML.replace("pytest tests/unit", "pytest tests/unit " + chr(124) * 2 + " exit 0")
        errors = sc.check_ci_not_neutered(self._tree(yml))
        self.assertTrue(any("masked" in e for e in errors), errors)

    def test_or_colon_in_ci_yml(self) -> None:
        yml = MIN_CI_YML.replace("pytest tests/unit", "pytest tests/unit " + chr(124) * 2 + " :")
        errors = sc.check_ci_not_neutered(self._tree(yml))
        self.assertTrue(any("masked" in e for e in errors), errors)

    def test_if_false_in_ci_yml(self) -> None:
        yml = MIN_CI_YML.replace("pytest tests/unit", "if false; then pytest tests/unit; fi")
        errors = sc.check_ci_not_neutered(self._tree(yml))
        self.assertTrue(any("masked" in e for e in errors), errors)

    def test_removed_self_check_job(self) -> None:
        yml = "name: CI\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n    - run: pytest tests/unit\n"
        errors = sc.check_ci_not_neutered(self._tree(yml))
        self.assertTrue(any("self-check job removed" in e for e in errors), errors)

    def test_disabled_self_check_job(self) -> None:
        yml = MIN_CI_YML.replace("  self-check:\n    runs-on:", "  self-check:\n    if: false\n    runs-on:")
        errors = sc.check_ci_not_neutered(self._tree(yml))
        self.assertTrue(any("disabled" in e for e in errors), errors)

    def test_gutted_main(self) -> None:
        errors = sc.check_ci_not_neutered(self._tree(MIN_CI_YML, "def main():\n    return 0\n"))
        self.assertTrue(any("gutted" in e or "stub" in e for e in errors), errors)

    def test_dockerfile_mask(self) -> None:
        root = self._tree(MIN_CI_YML)
        (root / "docker" / "Dockerfile.ci").write_text(
            "RUN uv pip install x " + chr(124) * 2 + " true\n", encoding="utf-8"
        )
        errors = sc.check_ci_not_neutered(root)
        self.assertTrue(any("Dockerfile.ci" in e for e in errors), errors)

    def test_pytest_k_filter_detected(self) -> None:
        self.assertEqual(sc.extract_pytest_filters("pytest tests/unit -k foo")[0], "foo")


class GitHelperTests(unittest.TestCase):
    def test_git_error_does_not_return_empty(self) -> None:
        with self.assertRaises(sc.GitError):
            sc.git("rev-parse", "this-ref-does-not-exist-zk0")


if __name__ == "__main__":
    unittest.main()
