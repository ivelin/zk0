"""Q-A gate probe: the config-equivalence probe must validate BOTH
registered optimizer-state policies against the current tree.

The study tree now carries (Q-A code change): reset must FAIL the probe
(state loads exist) and carry must PASS it (loads + wired save path).
"""
import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "config_equivalence_check", REPO / "study_prep" / "config_equivalence_check.py"
)
cec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cec)


def test_probe_rejects_reset_on_carry_tree():
    ok, _detail = cec.fl_optimizer_reset_probe("reset_per_round (waiver W-D11-ii)")
    assert ok is False


def test_probe_accepts_carry_on_carry_tree():
    ok, _detail = cec.fl_optimizer_reset_probe("carry")
    assert ok is True


def test_probe_rejects_unknown_policy():
    ok, _detail = cec.fl_optimizer_reset_probe("bogus-policy")
    assert ok is False
