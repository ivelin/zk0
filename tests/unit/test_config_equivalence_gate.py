"""Regression tests for the P1 config-equivalence gate.

Incident 1 of Study 001 (the camera miss, 2026-11-22): the FL server
template was built from a leftover two-camera dataset entry, so every
saved FL checkpoint declared front/top only while the design registered
three cameras. These tests pin the gate's core comparison logic so the
class of miss stays mechanically detectable. The full drill — checker
FAIL on the original Study 001 configuration, PASS on a corrected
miniature — is run by study_prep/config_equivalence_check.py and
recorded in the CI baseline report.
"""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "config_equivalence_check", REPO / "study_prep" / "config_equivalence_check.py"
)
cec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cec)

RENAME = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.top": "observation.images.camera2",
    "observation.images.wrist": "observation.images.camera3",
}
EXPECTED = {
    "observation.images.camera1": [3, 256, 256],
    "observation.images.camera2": [3, 256, 256],
    "observation.images.camera3": [3, 256, 256],
    "observation.state": [6],
}


def test_two_camera_template_does_not_equal_design():
    """The incident itself: front/top only must NOT match the design."""
    two_cam = {
        "observation.state": {"shape": [6]},
        "observation.images.front": {"shape": [3, 240, 320]},
        "observation.images.top": {"shape": [3, 240, 320]},
    }
    actual = cec.norm_features(cec.features_of_config({"input_features": two_cam}), RENAME)
    assert actual != EXPECTED
    assert "observation.images.camera3" not in actual


def test_three_camera_template_matches_design_after_rename():
    three_cam = {
        "observation.state": {"shape": [6]},
        "observation.images.front": {"shape": [3, 256, 256]},
        "observation.images.top": {"shape": [3, 256, 256]},
        "observation.images.wrist": {"shape": [3, 256, 256]},
    }
    actual = cec.norm_features(cec.features_of_config({"input_features": three_cam}), RENAME)
    assert actual == EXPECTED


def test_gate_verdict_fail_and_waiver():
    manifest = {}
    gate = cec.Gate(manifest, waivers={})
    gate.record("server_template.input_features", "fl-server", "f", False, EXPECTED, {})
    assert gate.verdict == "FAIL"

    waived = cec.Gate(manifest, waivers={"some.check": "W-TEST"})
    waived.record("some.check", "arm", "f", False, 1, 2)
    assert waived.checks[0]["status"] == "DIFF-WAIVED"
    assert waived.verdict == "PASS"


# ---- v2 (gate-fix build, audit F4/F6/F9) --------------------------------

NATIVE = {
    "observation.images.camera1": [3, 576, 1024],
    "observation.images.camera2": [3, 576, 1024],
    "observation.images.camera3": [3, 720, 1280],
    "observation.state": [6],
}


def test_constructed_rule_accepts_registered_or_native_shapes():
    assert cec.constructed_features_match(EXPECTED, EXPECTED, NATIVE)
    assert cec.constructed_features_match(NATIVE, EXPECTED, NATIVE)


def test_constructed_rule_rejects_wrong_resolution_right_names():
    """F6: the Incident-1 resolution with the right camera names must
    FAIL — 240x320 is neither the registered shape nor dataset-native."""
    wrong_res = {
        "observation.images.camera1": [3, 240, 320],
        "observation.images.camera2": [3, 240, 320],
        "observation.images.camera3": [3, 240, 320],
        "observation.state": [6],
    }
    assert not cec.constructed_features_match(wrong_res, EXPECTED, NATIVE)
    assert not cec.exact_features_match(wrong_res, EXPECTED)


def test_exact_rule_rejects_native_shapes_for_artifacts():
    """Artifacts of record are shape-exact: dataset-native declarations
    in a saved checkpoint do NOT pass (F6)."""
    assert not cec.exact_features_match(NATIVE, EXPECTED)
    assert cec.exact_features_match(EXPECTED, EXPECTED)


def test_budget_computation():
    """F4: steps x batch x views, from manifest inputs."""
    b = cec.compute_budget(15000, 32, 3)
    assert b == {"total_step_equivalents": 15000, "total_samples": 480000,
                 "images_at_3_views": 1440000}
    fl = cec.compute_budget(30 * 100 * 5, 32, 3)
    assert fl == b


def test_native_feature_shapes_from_manifest():
    manifest = {
        "expected_policy_config": {
            "rename_map": RENAME,
            "input_features": {"observation.state": [6]},
        },
        "dataset": {"cameras": {
            "observation.images.front": [576, 1024, 3],
            "observation.images.top": [576, 1024, 3],
            "observation.images.wrist": [720, 1280, 3],
        }},
    }
    assert cec.native_feature_shapes(manifest) == NATIVE
