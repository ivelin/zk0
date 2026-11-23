"""Regression tests for the P1 eval-side gate (audit F2).

Defect D9: the FL checkpoint declared two views while every other arm
declared three, and the eval fed each checkpoint its own declared
inputs. These tests pin the eval gate's core logic: the frozen-split
canonical hash input and the shape-exact feature identity rule.
"""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "eval_gate_check", REPO / "study_prep" / "eval_gate_check.py"
)
eg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eg)

EXPECTED = {
    "observation.images.camera1": [3, 256, 256],
    "observation.images.camera2": [3, 256, 256],
    "observation.images.camera3": [3, 256, 256],
    "observation.state": [6],
}
TWO_VIEW = {
    "observation.images.camera1": [3, 240, 320],
    "observation.images.camera2": [3, 240, 320],
    "observation.state": [6],
}


def test_two_view_checkpoint_fails_feature_identity():
    assert not eg.exact_features_match(TWO_VIEW, EXPECTED)


def test_canonical_split_rows_sorted_and_complete():
    split = {"tasks": {
        "b_task": {"episodes": [{"episode_index": 5, "length_frames": 10},
                                {"episode_index": 3, "length_frames": 7}]},
        "a_task": {"episodes": [{"episode_index": 1, "length_frames": 4}]},
    }}
    rows = eg.canonical_split_rows(split)
    assert rows == [["a_task", 1, 4], ["b_task", 3, 7], ["b_task", 5, 10]]
    # deterministic: same input -> identical hash input every time
    assert eg.canonical_split_rows(split) == rows
