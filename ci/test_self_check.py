#!/usr/bin/env python3
"""Harness for ci/self_check.py. Run: python3 ci/test_self_check.py"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    needs: self-check
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
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
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

    def test_semicolon_true(self) -> None:
        yml = MIN_CI_YML.replace("pytest tests/unit", "pytest tests/unit; true")
        errors = sc.check_ci_not_neutered(self._tree(yml))
        self.assertTrue(any("masked" in e for e in errors), errors)

    def test_dropped_needs_self_check(self) -> None:
        yml = MIN_CI_YML.replace("    needs: self-check\n", "")
        errors = sc.check_ci_not_neutered(self._tree(yml))
        self.assertTrue(any("dropped needs: self-check" in e for e in errors), errors)

    def test_poison_zk0_base_ci_dir(self) -> None:
        root = self._tree(MIN_CI_YML)
        _write(root, ".zk0-base-ci/subprocess.py", "raise SystemExit(0)\n")
        errors = sc.check_ci_not_neutered(root)
        self.assertTrue(any(".zk0-base-ci" in e for e in errors), errors)

    def test_pytest_k_filter_detected(self) -> None:
        self.assertEqual(sc.extract_pytest_filters("pytest tests/unit -k foo")[0], "foo")

    def test_n3_isolated_script_dir_not_on_sys_path(self) -> None:
        """N3/N4: isolated python must not import a sibling subprocess.py.

        N4: this test must be able to fail. The unisolated probe returns 99 when
        the sibling shadow is imported (the old SystemExit(0) poison could not
        fail). A workflow without ``python3 -I`` also fails the ci.yml assertion.
        """
        import subprocess as real_subprocess

        ci_yml = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
        workflow = ci_yml.read_text(encoding="utf-8")
        self.assertRegex(workflow, r"python3\s+-I\s+.+\bself_check\.py")
        broken_workflow = workflow.replace("python3 -I", "python3")
        self.assertIsNone(
            re.search(r"python3\s+-I\s+.+\bself_check\.py", broken_workflow)
        )

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script_dir = Path(tmp.name) / "base-ci"
        cwd = Path(tmp.name) / "cwd"
        script_dir.mkdir()
        cwd.mkdir()
        # Broken isolation imports this module and exits 99. The previous
        # poison used SystemExit(0), so an unisolated run still "passed".
        (script_dir / "subprocess.py").write_text(
            "raise SystemExit(99)\n", encoding="utf-8"
        )
        (script_dir / "probe.py").write_text(
            "import subprocess\n"
            "from pathlib import Path\n"
            "sib = Path(__file__).resolve().parent / 'subprocess.py'\n"
            "used = Path(subprocess.__file__).resolve()\n"
            "raise SystemExit(0 if used != sib else 2)\n",
            encoding="utf-8",
        )
        isolated = real_subprocess.run(
            [sys.executable, "-I", str(script_dir / "probe.py")],
            cwd=str(cwd),
            check=False,
        )
        self.assertEqual(isolated.returncode, 0)
        unisolated = real_subprocess.run(
            [sys.executable, str(script_dir / "probe.py")],
            cwd=str(cwd),
            check=False,
        )
        self.assertEqual(unisolated.returncode, 99)

    def test_detects_deleted_unit_test_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "tests/unit/test_kept.py", "def test_a():\n    pass\n")
            with mock.patch.object(sc, "base_ref", return_value="BASE"):
                with mock.patch.object(
                    sc,
                    "base_test_files",
                    return_value={
                        "tests/unit/test_kept.py": 1,
                        "tests/unit/test_gone.py": 2,
                    },
                ):
                    errors = sc.check_deleted_tests_and_count(root)
        self.assertTrue(
            any("deleted unit test file tests/unit/test_gone.py" in e for e in errors),
            errors,
        )
        self.assertTrue(any("collected test count dropped" in e for e in errors), errors)


class GitHelperTests(unittest.TestCase):
    def test_git_error_does_not_return_empty(self) -> None:
        with self.assertRaises(sc.GitError):
            sc.git("rev-parse", "this-ref-does-not-exist-zk0")

    def test_repo_root_prefers_zk0_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["ZK0_ROOT"] = tmp
            try:
                self.assertEqual(sc.repo_root(), Path(tmp).resolve())
            finally:
                os.environ.pop("ZK0_ROOT", None)


if __name__ == "__main__":
    unittest.main()
