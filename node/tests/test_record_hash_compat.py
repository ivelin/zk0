"""Regression tests for verifier fix F1 (slice-2 audit, 2026-11-17):
the content-hash rule must verify BOTH record eras end-to-end.

- Slice-1 era: `signature` is a status STRING, and it was inside
  the slice-1 hash body (benchmark.py still writes this form for
  hash-only records).
- Slice-2 era: `signature` is an Ed25519 signature BLOCK (a dict),
  excluded from the body it signs.

The slice-2 rule as first written excluded `signature`
unconditionally, so every slice-1 record failed `zk0 verify` with
"content hash: MISMATCH" even though the records were intact. The
rule now excludes `signature` only when it is a signature block,
and the CLI no longer treats a hash-only record's absent signature
as a verification failure.

The fixture is a real slice-1 record, copied byte-for-byte from the
node home that produced it: the certified Run B eval record
(sim1-realistic-20261117T203949Z, stored content hash 31508fd8...).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zk0_node.identity import NodeKey
from zk0_node.records import content_sha256, sign_record, verify_record

FIXTURE = (Path(__file__).resolve().parent / "fixtures"
           / "slice1_eval_record.json")
SLICE1_STORED_HASH = ("31508fd89e6bc9000852610e77a74dec4354b188c"
                      "732560e32ef576d953e9af7")
SLICE1_SIGNATURE_STRING = ("content hash only in build slice 1; "
                           "node-key signing lands with the Stage 1 "
                           "identity work")


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


class TestSlice1RecordCompat(unittest.TestCase):
    def test_fixture_is_the_real_slice1_record(self):
        rec = load_fixture()
        self.assertEqual(rec["schema"], "zk0-node-eval/0.1")
        self.assertEqual(rec["signature"], SLICE1_SIGNATURE_STRING)
        self.assertEqual(rec["content_sha256"], SLICE1_STORED_HASH)

    def test_slice1_hash_recomputes_identically(self):
        self.assertEqual(content_sha256(load_fixture()),
                         SLICE1_STORED_HASH)

    def test_slice1_record_verifies(self):
        v = verify_record(load_fixture())
        self.assertTrue(v["hash_ok"])
        self.assertFalse(v["signature_present"])

    def test_slice1_signature_string_is_hashed_content(self):
        # The string is inside the slice-1 hash body: editing it (or
        # any content) must break the hash — the compat rule must
        # not silently exclude it or fall back to a second rule.
        rec = load_fixture()
        rec["signature"] = "tampered"
        self.assertFalse(verify_record(rec)["hash_ok"])
        rec = load_fixture()
        rec["results"]["tier1"]["pass"] = False
        self.assertFalse(verify_record(rec)["hash_ok"])


class TestSlice2SignedRecordStillVerifies(unittest.TestCase):
    def test_signed_record_roundtrip_and_tamper(self):
        key = NodeKey(bytes(range(32)))
        rec = {"record_type": "t",
               "created_utc": "2026-11-17T00:00:00Z", "value": 42}
        rec["content_sha256"] = content_sha256(rec)
        rec = sign_record(rec, key)
        v = verify_record(rec)
        self.assertTrue(v["hash_ok"] and v["signature_ok"])
        rec["value"] = 43
        v2 = verify_record(rec)
        self.assertFalse(v2["hash_ok"] or v2["signature_ok"])


class TestCliVerifyEndToEnd(unittest.TestCase):
    def _run_cli_verify(self, path: Path):
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-m", "zk0_node.cli", "verify", str(path)],
            capture_output=True, text=True, env=env)

    def test_cli_accepts_slice1_record(self):
        r = self._run_cli_verify(FIXTURE)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("content hash: OK", r.stdout)

    def test_cli_rejects_tampered_slice1_record(self):
        rec = load_fixture()
        rec["results"]["tier1"]["pass"] = False
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tampered.json"
            p.write_text(json.dumps(rec))
            r = self._run_cli_verify(p)
        self.assertEqual(r.returncode, 1)
        self.assertIn("content hash: MISMATCH", r.stdout)


if __name__ == "__main__":
    unittest.main()
