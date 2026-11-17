"""Slice-2 unit tests: record signing (RFC 8032), the calibration
fingerprint, and the M0 join gate's refusal paths (Stage 3).

Stdlib unittest only, like the slice-1 tests. Gate tests use an
isolated ZK0_HOME; the eval check is disabled there (run_eval=False)
so these stay unit-fast — the full gate, eval included, is exercised
by the slice-2 integration runs.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zk0_node import calibration as C
from zk0_node.identity import NodeKey, load_or_create_key
from zk0_node.records import sign_record, verify_record


class TestEd25519(unittest.TestCase):
    def test_rfc8032_vector_1(self):
        # RFC 8032 section 7.1, TEST 1 (empty message).
        seed = bytes.fromhex(
            "9d61b19deffd5a60ba844af492ec2cc4"
            "4449c5697b326919703bac031cae7f60")
        key = NodeKey(seed)
        self.assertEqual(
            key.public_key_hex,
            "d75a980182b10ab7d54bfed3c964073a"
            "0ee172f3daa62325af021a68f707511a")
        sig = key.sign(b"")
        self.assertEqual(
            sig.hex(),
            "e5564300c360ac729086e2cc806e828a"
            "84877f1eb8e5d974d873e06522490155"
            "5fb8821590a33bacc61e39701cf9b46b"
            "d25bf5f0595bbe24655141438e7a100b")
        self.assertTrue(NodeKey.verify(key.public_key_hex, b"",
                                       sig.hex()))
        self.assertFalse(NodeKey.verify(key.public_key_hex, b"x",
                                        sig.hex()))

    def test_sign_verify_record_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            os.environ["ZK0_HOME"] = td
            try:
                key = load_or_create_key()
                rec = {"record_type": "t",
                       "created_utc": "2026-11-17T00:00:00Z", "value": 42}
                from zk0_node.records import content_sha256
                rec["content_sha256"] = content_sha256(rec)
                rec = sign_record(rec, key)
                v = verify_record(rec)
                self.assertTrue(v["hash_ok"] and v["signature_ok"])
                rec["value"] = 43
                v2 = verify_record(rec)
                self.assertFalse(v2["hash_ok"])
                self.assertFalse(v2["signature_ok"])
            finally:
                del os.environ["ZK0_HOME"]


class TestCalibrationFingerprint(unittest.TestCase):
    def _calib(self):
        return {"generation": 1, "offsets_deg": [0.0] * 6,
                "action_norm": {"type": "identity"}, "fingerprint": None}

    def test_fingerprint_stable_and_sensitive(self):
        c1 = self._calib()
        f1 = C.calibration_fingerprint(c1, "deadbeef")
        self.assertEqual(f1, C.calibration_fingerprint(self._calib(),
                                                       "deadbeef"))
        c2 = self._calib()
        c2["offsets_deg"][2] = 0.5
        self.assertNotEqual(f1, C.calibration_fingerprint(c2, "deadbeef"))
        c3 = self._calib()
        c3["action_norm"] = {"type": "per-arm", "stats_sha256": "x"}
        self.assertNotEqual(f1, C.calibration_fingerprint(c3, "deadbeef"))

    def test_identity_norm_roundtrip_is_exact(self):
        err = C.norm_roundtrip_max_err_deg(C.PROBE_POSES_DEG, self._calib())
        self.assertEqual(float(err), 0.0)


class TestJoinGateRefusals(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        os.environ["ZK0_HOME"] = self._td.name
        # The gate reads staged assets relative to ZK0_HOME; link the
        # real staged assets into the isolated home.
        real_assets = Path.home() / ".zk0-node" / "assets"
        if real_assets.exists():
            os.symlink(real_assets, Path(self._td.name) / "assets")
        from zk0_node.config import default_config, save_config
        self.cfg = default_config()
        save_config(self.cfg)

    def tearDown(self):
        del os.environ["ZK0_HOME"]
        self._td.cleanup()

    def _gate(self):
        from zk0_node.joingate import run_gate
        return run_gate(self.cfg, run_eval=False)

    def _arm_md5(self):
        from zk0_node.sim import assets
        return assets.arm_model_info("realistic").get("canonical_md5")

    def _good_calibration(self):
        from zk0_node.config import save_config
        calib = C.default_calibration()
        calib["generation"] = 1
        calib["fingerprint"] = C.calibration_fingerprint(
            calib, self._arm_md5())
        self.cfg["calibration"] = calib
        save_config(self.cfg)
        return calib

    def _save_calibration(self, calib):
        from zk0_node.config import save_config
        self.cfg["calibration"] = calib
        save_config(self.cfg)

    def test_skewed_calibration_refused_and_named(self):
        calib = self._good_calibration()
        calib["offsets_deg"][1] = 5.0  # deliberately corrupted probe data
        calib["fingerprint"] = C.calibration_fingerprint(
            calib, self._arm_md5())
        self._save_calibration(calib)
        rec = self._gate()
        self.assertFalse(rec["pass"])
        self.assertIn("calibration_standardization", rec["failed_checks"])

    def test_per_arm_normalization_refused_and_named(self):
        calib = self._good_calibration()
        calib["action_norm"] = {"type": "per-arm", "stats_sha256": "abc"}
        calib["fingerprint"] = C.calibration_fingerprint(
            calib, self._arm_md5())
        self._save_calibration(calib)
        rec = self._gate()
        self.assertFalse(rec["pass"])
        self.assertIn("action_normalization", rec["failed_checks"])

    def test_stale_fingerprint_refused_and_named(self):
        calib = self._good_calibration()
        calib["offsets_deg"][0] = 0.2   # silent edit after fingerprinting
        self._save_calibration(calib)
        rec = self._gate()
        self.assertFalse(rec["pass"])
        self.assertIn("calibration_fingerprint_current",
                      rec["failed_checks"])

    def test_require_attestation_refuses_without_one(self):
        from zk0_node.joingate import GateRefused, require_current_attestation
        with self.assertRaises(GateRefused):
            require_current_attestation(self.cfg)


if __name__ == "__main__":
    unittest.main()
