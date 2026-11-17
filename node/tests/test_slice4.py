"""Slice-4 unit tests: the Stage 7 envelope + admission
bookkeeping rules, the Stage 10/11 comparison + declaration
arithmetic (pure functions over fabricated records), the FL
client's identity/no-op dry-run mode, and the vendored hardened
pilot's identity (calibration bands + law parameters).

The sim-heavy paths (admission runs, fine-tune, suites, the round)
are exercised by the slice-4 integration runs and the full-loop
dry run, not here — these tests pin the rules those runs apply.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zk0_node.flow import (build_comparison, declaration_verdict,
                           envelope_declaration, envelope_draws,
                           marks_block, render_declaration_text)


def suite_record(task, k, n=20):
    return {"task": {"id": task},
            "results": {"tier1": {"successes": k, "trials": n,
                                  "rate": k / n}},
            "content_sha256": f"fake-{task}-{k}"}


def round_record(accepted=("node-a",)):
    return {"participants": [{"accepted_node_ids": list(accepted)}]}


class TestEnvelope(unittest.TestCase):
    def test_seven_draws_corners_centre_interior(self):
        draws = envelope_draws("sim2")
        self.assertEqual(len(draws), 7)
        ids = [d["draw_id"] for d in draws]
        self.assertEqual(ids[:4], ["corner-mu_lo-mass_lo",
                                   "corner-mu_lo-mass_hi",
                                   "corner-mu_hi-mass_lo",
                                   "corner-mu_hi-mass_hi"])
        self.assertEqual(ids[4], "centre")
        corners = {(d["mu"], d["mass_kg"]) for d in draws[:4]}
        self.assertEqual(corners, {(0.12, 0.1062), (0.12, 0.1298),
                                   (0.35, 0.1062), (0.35, 0.1298)})
        centre = draws[4]
        self.assertEqual((centre["mu"], centre["mass_kg"]),
                         (0.235, 0.118))
        for d in draws[5:]:
            self.assertTrue(0.12 <= d["mu"] <= 0.35)
            self.assertTrue(0.1062 <= d["mass_kg"] <= 0.1298)

    def test_sim1_masses(self):
        draws = envelope_draws("sim1")
        masses = {d["mass_kg"] for d in draws[:5]}
        self.assertEqual(masses, {0.0544, 0.0736, 0.064})

    def test_deterministic_and_never_nominal_alone(self):
        self.assertEqual(envelope_draws("sim1"), envelope_draws("sim1"))
        pts = {(d["mu"], d["mass_kg"]) for d in envelope_draws("sim1")}
        self.assertGreater(len(pts), 1)
        self.assertNotEqual(pts, {(0.35, 0.064)})

    def test_declaration_carries_decision52_fields(self):
        env = envelope_declaration()
        fields = env["fields"]
        for f in ("friction_pair_mu", "mass_kg", "servo_class",
                  "fragility"):
            self.assertIn(f, fields)
        self.assertIn("MEASURED", fields["mass_kg"]["sim2"]["tag"])


class TestMarks(unittest.TestCase):
    def test_three_axes_never_merged(self):
        m = marks_block("sim-trained", "ORACLE-GUIDED",
                        {"envelope_id": "x"})
        self.assertEqual(set(m), {"provenance", "generation",
                                  "envelope"})
        self.assertEqual(m["provenance"], "sim-trained")
        self.assertEqual(m["generation"], "ORACLE-GUIDED")


class TestComparisonAndDeclaration(unittest.TestCase):
    def _comparison(self, pre, bench, post):
        return build_comparison(
            {"SIM-1": suite_record("SIM-1", pre[0]),
             "SIM-2": suite_record("SIM-2", pre[1])},
            {"SIM-1": suite_record("SIM-1", bench[0]),
             "SIM-2": suite_record("SIM-2", bench[1])},
            {"SIM-1": suite_record("SIM-1", post[0]),
             "SIM-2": suite_record("SIM-2", post[1])})

    def test_win_when_parity_and_no_self_regression(self):
        cmp_ = self._comparison((0, 0), (8, 6), (8, 6))
        v = declaration_verdict(cmp_, round_record())
        self.assertTrue(v["win"])
        self.assertTrue(v["criteria"]["parity_with_tolerance"]["pass"])
        self.assertTrue(v["criteria"]["no_self_regression"]["pass"])
        # Fleet regression is disclosed as not evaluable, never
        # silently passed — and does not fabricate a win blocker.
        self.assertIsNone(
            v["criteria"]["no_fleet_regression"]["pass"])

    def test_parity_tolerance_edge(self):
        # post exactly 5pp below benchmark passes; 10pp fails.
        cmp_ = self._comparison((0, 0), (10, 10), (9, 9))
        self.assertTrue(all(t["parity"]["pass"]
                            for t in cmp_["tasks"].values()))
        cmp_ = self._comparison((0, 0), (10, 10), (8, 8))
        v = declaration_verdict(cmp_, round_record())
        self.assertFalse(v["win"])
        self.assertFalse(v["criteria"]["parity_with_tolerance"]["pass"])

    def test_self_regression_blocks_win(self):
        cmp_ = self._comparison((10, 10), (4, 4), (9, 9))
        v = declaration_verdict(cmp_, round_record())
        self.assertFalse(v["win"])
        self.assertFalse(v["criteria"]["no_self_regression"]["pass"])

    def test_no_round_record_no_win(self):
        cmp_ = self._comparison((0, 0), (8, 8), (8, 8))
        v = declaration_verdict(cmp_, None)
        self.assertFalse(v["win"])
        self.assertFalse(
            v["criteria"]["contribution_accepted"]["pass"])

    def test_declaration_text_is_generated_from_numbers(self):
        cmp_ = self._comparison((0, 0), (8, 6), (6, 6))
        v = declaration_verdict(cmp_, round_record())
        text = render_declaration_text(cmp_, v)
        self.assertIn("VERDICT: NO WIN", text)  # SIM-1 parity fails
        self.assertIn("8/20 (40%)", text)
        self.assertIn("6/20 (30%)", text)
        self.assertIn("NOT EVALUABLE", text)
        self.assertIn("PENDING", text)          # preregistration label
        self.assertIn("identity/no-op", text)   # dry-run scope label
        self.assertIn("+10pp", text)            # section 5.4 limit

    def test_win_text(self):
        cmp_ = self._comparison((0, 0), (8, 6), (8, 6))
        v = declaration_verdict(cmp_, round_record())
        self.assertIn("VERDICT: WIN",
                      render_declaration_text(cmp_, v))


class TestNoopClient(unittest.TestCase):
    def test_identity_update(self):
        import numpy as np
        import torch

        from zk0_node.fl.client import Zk0NumPyClient
        from zk0_node.fl.common import arrays_hash, get_subset
        from zk0_node.identity import NodeKey

        torch.manual_seed(0)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.expert = torch.nn.Linear(4, 3)
                self.other = torch.nn.Linear(4, 4)

        policy = Tiny()
        key = NodeKey(bytes(range(32)))
        client = Zk0NumPyClient(policy, None, None, "node-b", key,
                                noop=True)
        self.assertEqual(client.n_examples, 0)
        params = get_subset(policy)
        out, n, metrics = client.fit(params, {"steps": 400})
        self.assertEqual(n, 0)
        for a, b in zip(out, params):
            self.assertIs(a, b)  # literally the same arrays back
        self.assertEqual(metrics["update_hash"], arrays_hash(params))
        self.assertIn("identity-noop", metrics["update_kind"])
        self.assertTrue(NodeKey.verify(
            key.public_key_hex, bytes.fromhex(metrics["update_hash"]),
            metrics["update_sig"]))


class TestVendoredHardened(unittest.TestCase):
    def test_calibration_bands_are_the_certified_ones(self):
        import json
        cal = json.loads((Path(__file__).resolve().parents[1]
                          / "src" / "zk0_node" / "sim" / "hardened"
                          / "calibration.json").read_text())
        self.assertEqual((cal["rim"]["d_lo"], cal["rim"]["d_hi"]),
                         (0.1515, 0.1759))
        self.assertEqual((cal["pinch"]["d_lo"], cal["pinch"]["d_hi"]),
                         (0.1337, 0.1715))

    def test_hardened_law_parameters(self):
        src = (Path(__file__).resolve().parents[1] / "src"
               / "zk0_node" / "sim" / "hardened"
               / "hardened_pilot.py").read_text()
        self.assertIn("class HardenedRealisticController", src)
        self.assertIn("N_C = 9", src)          # 9-tick persistence
        self.assertIn("SIG_DUTY = 0.01", src)  # placeholder noise
        self.assertIn("SIG_Q = 0.005", src)
        prov = (Path(__file__).resolve().parents[1] / "src"
                / "zk0_node" / "sim" / "hardened"
                / "PROVENANCE.md").read_text()
        self.assertIn("ebf1d1a3671ac9de4ee0946222925cec", prov)


if __name__ == "__main__":
    unittest.main()
