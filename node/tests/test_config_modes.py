"""Mode/gate unit tests (spec §2, §2.1, §2.3). Stdlib unittest only —
no test dependencies are added to the runtime venv."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zk0_node.config import (ProdEnrollmentBlocked, default_config,
                             enroll_prod, prod_block_reasons, set_local_scope,
                             set_mode)
from zk0_node.safety import DEFAULT_LIMITS, clamp_command, workspace_contains


class TestProdHardBlock(unittest.TestCase):
    def test_external_default_is_blocked(self):
        cfg = default_config()
        reasons = prod_block_reasons(cfg)
        self.assertTrue(any("hard-blocked" in r for r in reasons))
        self.assertTrue(any("acceptance" in r for r in reasons))
        with self.assertRaises(ProdEnrollmentBlocked):
            enroll_prod(cfg)

    def test_external_with_acceptance_but_terms_not_live_still_blocked(self):
        cfg = default_config()
        cfg["terms"]["acceptance_record"] = "acc-test-1"
        with self.assertRaises(ProdEnrollmentBlocked):
            enroll_prod(cfg)

    def test_external_terms_live_plus_acceptance_allowed(self):
        cfg = default_config()
        cfg["terms"]["live"] = True
        cfg["terms"]["acceptance_record"] = "acc-test-1"
        rec = enroll_prod(cfg)
        self.assertEqual(rec["scope"], "prod")

    def test_founder_with_own_acceptance_record_allowed(self):
        cfg = default_config()
        cfg["identity_kind"] = "founder"
        cfg["terms"]["acceptance_record"] = "acc-founder-1"
        self.assertEqual(prod_block_reasons(cfg), [])
        self.assertEqual(enroll_prod(cfg)["identity_kind"], "founder")

    def test_founder_without_acceptance_record_blocked(self):
        cfg = default_config()
        cfg["identity_kind"] = "founder"
        with self.assertRaises(ProdEnrollmentBlocked):
            enroll_prod(cfg)


class TestModeSwitching(unittest.TestCase):
    def test_up_to_prod_blocked_for_external(self):
        cfg = default_config()
        with self.assertRaises(ProdEnrollmentBlocked):
            set_mode(cfg, "prod")
        self.assertEqual(cfg["mode"], "sim")  # unchanged on refusal

    def test_down_switch_always_allowed(self):
        cfg = default_config()
        cfg["mode"] = "prod"
        notes = set_mode(cfg, "sim")
        self.assertEqual(cfg["mode"], "sim")
        self.assertTrue(any("down-switch" in n for n in notes))

    def test_up_to_staging_allowed_and_scoped(self):
        cfg = default_config()
        set_mode(cfg, "staging")
        self.assertEqual(cfg["mode"], "staging")
        self.assertIn("staging", cfg["network_scopes"])

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            set_mode(default_config(), "moon")

    def test_local_scope_is_orthogonal(self):
        cfg = default_config()
        set_local_scope(cfg, True)
        self.assertIn("local", cfg["network_scopes"])
        self.assertEqual(cfg["mode"], "sim")
        set_local_scope(cfg, False)
        self.assertNotIn("local", cfg["network_scopes"])


class TestSafetyClamp(unittest.TestCase):
    def test_s6b_command_clamped_to_joint_floor(self):
        grip_range = (-0.1745, 1.048)  # so101-realistic gripper range (rad)
        applied, clamped = clamp_command(-0.65, grip_range)
        self.assertTrue(clamped)
        self.assertEqual(applied, grip_range[0])

    def test_in_range_command_untouched(self):
        applied, clamped = clamp_command(0.5, (-0.1745, 1.048))
        self.assertFalse(clamped)
        self.assertEqual(applied, 0.5)

    def test_workspace_contains_sim1_scene(self):
        self.assertTrue(workspace_contains(DEFAULT_LIMITS, (0.26, 0.02, 0.0205)))
        self.assertTrue(workspace_contains(DEFAULT_LIMITS, (0.27, -0.14, 0.035)))
        self.assertFalse(workspace_contains(DEFAULT_LIMITS, (1.5, 0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
